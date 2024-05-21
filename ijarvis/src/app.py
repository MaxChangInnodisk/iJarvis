import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import utils
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Form,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from logger import iLogger
from params import Process, llm_conf, nlp_conf, shared, stt_conf, vlm_conf
from thirdparty import core, llm, nlp, stt, vlm
from utils import SharedKiller
from v4l2 import camera

CAM1 = "/dev/video0"
CAM2 = "/dev/video2"
CAM2 = "rtsp://admin:123456@172.16.93.104:1184/stream0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    iLogger.info("Starting lifespan setup")
    skiller = SharedKiller()
    shared.exec = core.AsyncExecutor(
        loop=asyncio.get_event_loop(), executor=ThreadPoolExecutor(max_workers=4)
    )
    results = await asyncio.gather(
        shared.exec(stt.AsyncWhisper, stt_conf.name, shared.exec),
        shared.exec(llm.Mistral, llm_conf.host, llm_conf.port, llm_conf.route),
        shared.exec(
            nlp.NLTK,
            nlp_conf.path,
        ),
        shared.exec(
            vlm.GroundingDino,
            vlm_conf.host,
            vlm_conf.port,
            vlm_conf.route,
            nlp_conf.path,
        ),
        shared.exec(camera.CameraStream, CAM1),
        shared.exec(camera.CameraStream, CAM2),
    )
    (
        shared.stt,
        shared.llm,
        shared.nlp,
        shared.vlm,
        shared.cams["0"],
        shared.cams["1"],
    ) = results

    yield

    iLogger.warning("Starting lifespan teardown")
    skiller.release()
    shared.cams["0"].release()
    shared.cams["1"].release()
    shared.stt.release()
    iLogger.warning("Resources released")


app = FastAPI(lifespan=lifespan)


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/process")
async def get_process(process_uuid: str = ""):
    if process_uuid == "":
        return JSONResponse(
            content={uuid: proc.model_dump() for uuid, proc in shared.proc.items()}
        )
    if process_uuid not in shared.proc:
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Not supported process uuid",
        )
    return JSONResponse(content=shared.proc[process_uuid].model_dump())


@app.post("/stt")
async def speech_to_text(
    background_tasks: BackgroundTasks, audio: UploadFile = UploadFile(...)
):
    if audio is None:
        raise HTTPException(status_code=400, detail="No file uploaded or empty content")
    process_uuid = str(uuid.uuid4())
    shared.proc[process_uuid] = Process(
        type=stt_conf.type, message="Start process WAV file"
    )
    audio_file_path = Path(process_uuid + ".wav")
    with open(audio_file_path, "wb") as f:
        f.write(await audio.read())
    background_tasks.add_task(process_stt, process_uuid, audio_file_path)
    return JSONResponse(content={"process_uuid": process_uuid})


async def process_stt(process_uuid: str, audio_path: Path, remove_audio: bool = True):
    ts = time.time()
    content = await shared.stt.inference(audio_path)
    tc = time.time() - ts
    if remove_audio:
        audio_path.unlink()
    shared.proc[process_uuid].status = "done"
    shared.proc[process_uuid].message = content
    shared.proc[process_uuid].performance["inference"] = tc


@app.post("/image2string")
async def upload_file(file: UploadFile = UploadFile(...)):
    try:
        file_data = utils.buffer_to_cv(await file.read())
        file_base64 = utils.cv_to_str(file_data)

        return JSONResponse(
            content={
                "data": file_base64,
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    image: str = Form(...),
    prompt: str = Form(...),
    box_color: Optional[str] = Form(default=None),
    text_color: Optional[str] = Form(default=None),
    box_threshold: Optional[str] = Form(default=None),
    text_threshold: Optional[str] = Form(default=None),
):
    process_uuid = str(uuid.uuid4())
    shared.proc[process_uuid] = Process(
        message="Start predict", created_time=time.time()
    )

    ret, out = nlp.is_setting_event(nlp_model=shared.nlp, prompt=prompt)
    if ret:
        title = ["Got it", "Roger that", "No problem", "Sure", "Okay"]
        if ret == "threshold":
            shared.option.box_threshold = out
            iLogger.warning(f"[SET] {ret} -> {shared.option.box_threshold}")
        elif ret == "color":
            shared.option.box_color = out
            iLogger.warning(f"[SET] {ret} -> {shared.option.box_color}")
        shared.proc[process_uuid] = Process(
            message=f"{title[int(time.time())%5]}. Set {ret} to {out}", status="done"
        )
        return JSONResponse(content={"process_uuid": process_uuid})

    cur_box_threshold = box_threshold if box_threshold else shared.option.box_threshold
    cur_box_color = box_color if box_color else shared.option.box_color
    cur_text_threshold = (
        text_threshold if text_threshold else shared.option.text_threshold
    )
    cur_text_color = text_color if text_color else shared.option.text_color
    if isinstance(cur_box_color, str):
        cur_box_color = json.loads(cur_box_color)
    if isinstance(cur_text_color, str):
        cur_text_color = json.loads(cur_text_color)

    drawer = vlm.GroundingDinoDrawer(
        box_color=cur_box_color,
        font_color=cur_text_color,
        alpha=float(shared.option.alpha),
        fill_box=int(shared.option.fill_box),
    )
    # Ensure the base64 content is correct
    if "," in image:
        image = image.split(",")[1]
    # Add background task
    background_tasks.add_task(
        process_predict_drawer,
        process_uuid,
        image,
        prompt,
        drawer,
        float(cur_box_threshold),
        float(cur_text_threshold),
    )
    return JSONResponse(content={"process_uuid": process_uuid})


async def process_predict_drawer(
    process_uuid: str,
    image: str,
    prompt: str,
    drawer: vlm.GroundingDinoDrawer,
    box_threshold: float,
    text_threshold: float,
):
    shared.proc[process_uuid].status = "doing"
    if "?" not in prompt:
        prompt += "?"

    if "where" in prompt.lower():
        shared.proc[process_uuid].type = vlm_conf.type
        ts = time.time()
        results = await shared.vlm.inference(
            image, prompt, box_threshold, text_threshold
        )
        iLogger.info(results)
        tc = time.time() - ts
        if results and isinstance(results, list):
            draw_image = drawer.draw(image=utils.str_to_cv(image), results=results)
            shared.proc[process_uuid].status = "done"
            shared.proc[
                process_uuid
            ].message = f"found {len(results)} {results[0].label}"
            shared.proc[process_uuid].image = utils.cv_to_str(draw_image)
            shared.proc[process_uuid].performance["inference"] = tc
        else:
            shared.proc[process_uuid].status = "done"
            shared.proc[process_uuid].message = "not found"
    else:
        shared.proc[process_uuid].type = llm_conf.type
        ts = time.time()
        results = await shared.llm.inference(content=prompt)
        tc = time.time() - ts
        shared.proc[process_uuid].status = "done"
        shared.proc[process_uuid].message = results
        shared.proc[process_uuid].performance["inference"] = tc
    # else:
    #     shared.proc[process_uuid].status = "error"
    #     shared.proc[process_uuid].message = "Must have `where` or `what`."


@app.websocket("/websocket/stt")
async def websocket_stt(
    websocket: WebSocket, process_uuid: str = Query(..., description="Process UUID")
):
    await websocket.accept()
    try:
        if process_uuid not in shared.proc:
            process_error = Process(
                status="error", message="not supported uuid"
            ).model_dump_json()
            await websocket.send_text(process_error)
            raise KeyError(process_error["message"])

        while True:
            await websocket.send_text(shared.proc[process_uuid].model_dump_json())
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        iLogger.warning(f"Unexpected error: {e}")
    finally:
        await websocket.close()


@app.websocket("/websocket/predict")
async def websocket_predict(
    websocket: WebSocket, process_uuid: str = Query(..., description="Process UUID")
):
    await websocket.accept()
    try:
        while True:
            if process_uuid not in shared.proc:
                process_error = Process(
                    status="error", message="not supported uuid"
                ).model_dump_json()
                await websocket.send_text(process_error)
                raise KeyError(process_error["message"])

            await websocket.send_text(shared.proc[process_uuid].model_dump_json())
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        # iLogger.warning(f"WebSocket disconnected: index {index}")
        pass
    except Exception as e:
        iLogger.warning(f"Unexpected error: {e}")
    finally:
        await websocket.close()


@app.websocket("/camera")
async def camera_endpoint(
    websocket: WebSocket, index: str = Query(..., description="Camera index")
):
    await websocket.accept()

    try:
        if index not in shared.cams:
            cams = shared.cams.keys()
            msg = f"Expect index is {', '.join(cams)}"
            await websocket.send_text(msg)
            raise KeyError(msg)
        while True:
            frame = shared.cams[index].read()
            if frame is not None:
                await websocket.send_bytes(utils.cv_to_buffer(frame))
                await asyncio.sleep(0)
            await asyncio.sleep(1 / shared.cams[index].fps)
    except WebSocketDisconnect:
        # iLogger.warning(f"WebSocket disconnected: index {index}")
        pass
    except Exception as e:
        iLogger.warning(f"Unexpected error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=9527)
