from pathlib import Path

import fal
import PIL.Image
from fal.toolkit import download_model_weights
from fastapi import HTTPException


def patch_onnx_runtime(
    inter_op_num_threads: int = 16,
    intra_op_num_threads: int = 16,
    omp_num_threads: int = 16,
):
    import os

    import onnxruntime as ort

    os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    _default_session_options = ort.capi._pybind_state.get_default_session_options()

    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = inter_op_num_threads
        _default_session_options.intra_op_num_threads = intra_op_num_threads
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new


def draw_kps(
    image_pil,
    kps,
    color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)],
):
    import math

    import cv2
    import numpy as np
    import PIL.Image

    stickwidth = 4
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    if isinstance(image_pil, PIL.Image.Image):
        out_img = np.array(image_pil)
    else:
        out_img = image_pil

    for i, _ in enumerate(limb_seq):
        index = limb_seq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def checksum(
    obj,
    method="md5",
    *,
    # If dicts with different key order are to be treated as identical.
    # Set False otherwise.
    sort_keys=True,
    # A dict with circular referencing objects can never be hashed and is
    # out of scope of this topic. Skip checking such cases to save an
    # in-memory mapping as we don't expect them. Set True otherwise.
    check_circular=False,
    # Set True to output bytes instead of hex string to save memory, if the
    # checksum is used only for internal comparison and not to be output.
    ouput_bytes=False,
):
    import hashlib
    import json

    m = hashlib.new(method)
    encoder = json.JSONEncoder(
        check_circular=check_circular,
        sort_keys=sort_keys,
        ensure_ascii=False,  # don't escape Unicode chars to save bytes
        separators=(",", ":"),  # reduce default spaces to be more compact
    )
    for chunk in encoder.iterencode(obj):
        m.update(chunk.encode("UTF-8"))

    if ouput_bytes:
        return m.digest()

    return m.hexdigest()


class Preprocessor:
    def __init__(self):
        patch_onnx_runtime()

    @fal.cached
    def load_model(self, processor_id: str) -> None:
        from controlnet_aux.processor import Processor

        return Processor(
            processor_id, {"detect_resolution": 1024, "image_resolution": 1024}
        )

    @fal.cached
    def load_face_detection_model(self, model_url) -> None:

        import os

        import insightface.utils.storage
        from insightface.app import FaceAnalysis

        insightface.utils.storage.BASE_REPO_URL = (
            "https://storage.googleapis.com/fal-models/insightface"
        )

        os.environ["HF_HOME"] = "/data/models"

        patch_onnx_runtime()

        face_detection_model_name = None
        try:
            if model_url.startswith("https://"):
                print("Assuming face_detection model path is a URL")
                face_detection_path = download_model_weights(
                    model_url.insight_face_model_path,
                )
                face_detection_model_dir = Path(face_detection_path).parent
                face_detection_model_name = Path(face_detection_path).name
            elif model_url.startswith("http://"):
                # raise an error if the path is an http link
                raise HTTPException(
                    422,
                    detail="HTTP links are not supported for face_detection model weights. Please use HTTPS links or local paths.",
                )
            # see if there is a single forward slash in the path
            elif model_url.count("/") == 1:
                raise HTTPException(
                    422,
                    detail="Huggingface models for face_detection are not supported!",
                )
            else:
                # assume it is a model name
                face_detection_model_name = model_url
                face_detection_model_dir = None

        except Exception as e:
            raise HTTPException(
                422,
                detail=f"Failed to download face_detection model: {e}",
            )

        if face_detection_model_dir is None:
            app = FaceAnalysis(
                name=face_detection_model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        else:
            app = FaceAnalysis(
                name=face_detection_model_name,
                root=face_detection_model_dir,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        return app

    def run_face_detection(
        self,
        image,
        model,
        det_size_width: int = 640,
        det_size_height: int = 640,
        max_face_num: int = 1,
        sorting: str = "largest-to-smallest",
    ) -> dict:
        import os

        import cv2
        import numpy as np
        import torch

        output_dir = "/data/face_embeddings/"
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            model.prepare(ctx_id=0, det_size=(det_size_width, det_size_height))

            cv2_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
            faces = model.get(cv2_image)

            if len(faces) == 0:
                raise HTTPException(
                    422,
                    detail="No faces detected",
                )

            face_detections = []
            zero_image = np.zeros(
                (cv2_image.shape[0], cv2_image.shape[1], 3), dtype=np.uint8
            )
            for face in faces:
                kps_image = draw_kps(zero_image, face.kps)

                face_detections.append(
                    {
                        "bbox": face.bbox,
                        "kps": face.kps,
                        "kps_image": kps_image,
                        "det_score": face.det_score,
                        "embedding": face.embedding,
                        "sex": face.sex,
                    }
                )

            if sorting == "largest-to-smallest":
                face_detections = sorted(
                    face_detections, key=lambda x: -x["bbox"][2] * x["bbox"][3]
                )
            elif sorting == "smallest-to-largest":
                face_detections = sorted(
                    face_detections, key=lambda x: x["bbox"][2] * x["bbox"][3]
                )
            elif sorting == "left-to-right":
                face_detections = sorted(face_detections, key=lambda x: x["bbox"][0])
            elif sorting == "right-to-left":
                face_detections = sorted(face_detections, key=lambda x: -x["bbox"][0])

            top_face = face_detections[0]
            face_detections = face_detections[:max_face_num]

        output = {
            "faces": face_detections,
            "top_face": top_face,
        }
        return output

    def process(self, image: PIL.Image.Image, processor_id) -> PIL.Image.Image:
        if processor_id == "face_kps":
            processor = self.load_face_detection_model("antelopev2")
            processed_image = self.run_face_detection(image, processor)["top_face"]
        else:
            processor = self.load_model(processor_id)
            processed_image = processor(image, to_pil=True)

        return processed_image
