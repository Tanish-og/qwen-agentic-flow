import time
import operator
import io
import base64
import json
import re
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from langgraph.graph import StateGraph, END

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from ultralytics import YOLO
import numpy as np

import streamlit as st


class CropResult(BaseModel):
    crop: str = Field(description="Crop name from the valid options list")

class PestResult(BaseModel):
    name: str         = Field(description="Pest name from valid options or 'none'")
    type: str         = Field(description="Always 'pest'")
    present: bool     = Field(description="True if live insect visible")
    confidence: float = Field(description="0.0-1.0")
    bbox: List[int]   = Field(default=[0,0,0,0], description="Tight bounding box [ymin,xmin,ymax,xmax] in 0-1000 scale. Top-left=(0,0), bottom-right=(1000,1000). Must be tight around the pest only, not the whole image.")

class SymptomResult(BaseModel):
    name: str         = Field(description="Pest name responsible for damage or 'none'")
    type: str         = Field(description="Always 'symptom'")
    present: bool     = Field(description="True if damage holes/bites visible")
    confidence: float = Field(description="0.0-1.0")
    bbox: List[int]   = Field(default=[0,0,0,0], description="Tight bounding box [ymin,xmin,ymax,xmax] in 0-1000 scale. Top-left=(0,0), bottom-right=(1000,1000). Must be tight around the damaged region only.")

class DiseaseResult(BaseModel):
    name: str         = Field(description="Disease name from valid options or 'none'")
    type: str         = Field(description="Always 'disease'")
    present: bool     = Field(description="True if disease spots/mold/blight visible")
    confidence: float = Field(description="0.0-1.0")
    bbox: List[int]   = Field(default=[0,0,0,0], description="Tight bounding box [ymin,xmin,ymax,xmax] in 0-1000 scale. Top-left=(0,0), bottom-right=(1000,1000). Must be tight around the most prominent diseased area only.")


@dataclass
class CropKnowledge:
    crop_name: str
    pests:     List[str] = field(default_factory=list)
    diseases:  List[str] = field(default_factory=list)
    aliases:   List[str] = field(default_factory=list)


class AgriculturalKnowledgeBase:
    def __init__(self):
        self.crops = {
            "paddy":  CropKnowledge("paddy",  ["rice leaf folder","yellow stem borer","brown plant hopper"], ["bacterial leaf blight","false smut","sheath blight"], ["rice"]),
            "maize":  CropKnowledge("maize",  ["fall armyworm"],                                             ["common rust"],                                        ["corn"]),
            "cotton": CropKnowledge("cotton", ["whitefly","thrips","aphids","pink bollworm"],                ["bacterial blight","alternaria leaf spot"]),
            "tomato": CropKnowledge("tomato", ["whitefly","fruit borer"],                                    ["early blight","late blight","leaf curl virus"]),
            "banana": CropKnowledge("banana", ["red spider mite","banana leaf roller"],                           ["banana sigatoka leafspot"]),
        }

    def get_crop_knowledge(self, name: str) -> Optional[CropKnowledge]:
        name = name.lower().strip()
        if name in self.crops: return self.crops[name]
        for c in self.crops.values():
            if name in c.aliases: return c
        return None


class ModelRepository:
    def __init__(self):
        self._paths: Dict[str, Dict[str, str]] = {
            "paddy": {
                "pest":    "yolo_weights/paddy_pest.pt",
                "disease": "yolo_weights/paddy_disease.pt",
            },
            "maize": {
                "pest":    "yolo_weights/maize_pest.pt",
                "disease": None,
            },
            "cotton": {
                "pest":    "yolo_weights/cotton_pest.pt",
                "disease": None,
            },
            "banana": {
                "pest":    "yolo_weights/banana_pest.pt",
                "disease": "yolo_weights/banana_disease.pt",
            },
        }
        self._cache: Dict[str, YOLO] = {}

    def get_model(self, crop: str, category: str) -> Optional[YOLO]:
        if category == "symptom":
            category = "pest"
        key = f"{crop}_{category}"
        if key in self._cache:
            return self._cache[key]
        path = self._paths.get(crop, {}).get(category)
        if not path:
            print(f"[ModelRepository] No model registered for {crop}/{category}")
            return None
        try:
            model = YOLO(path)
            self._cache[key] = model
            print(f"[ModelRepository] Loaded {key} from {path}")
            return model
        except Exception as e:
            print(f"[ModelRepository] Failed to load {key}: {e}")
            return None


class AgentState(TypedDict):
    image_obj:            Image.Image
    crop_name:            str
    crop_knowledge:       Optional[CropKnowledge]
    detections:           Annotated[List[Dict[str, Any]], operator.add]
    confidence_threshold: float
    status:               str
    error:                Optional[str]
    encoded_inputs:       Optional[Dict]
    yolo_detections:      List[Dict[str, Any]]


SYSTEM = (
    "You are an Agricultural AI. Analyze the image carefully. "
    "Reply ONLY with a valid JSON object. No markdown fences, no explanation. "
    "For bbox: look at where the target actually appears in the image. "
    "Coordinates are 0-1000 where (0,0)=top-left and (1000,1000)=bottom-right. "
    "Be PRECISE — do not default to center [250,250,750,750]."
)

VLLM_HOST  = st.secrets.get("VLLM_HOST", "http://localhost:8000")
VLLM_MODEL = st.secrets.get("VLLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")


class AgriInferenceEngine:
    def __init__(self, vllm_host: str = VLLM_HOST, model_name: str = VLLM_MODEL):
        self.client     = OpenAI(base_url=f"{vllm_host}/v1", api_key="EMPTY")
        self.model_name = model_name

        self.crop_parser    = JsonOutputParser(pydantic_object=CropResult)
        self.pest_parser    = JsonOutputParser(pydantic_object=PestResult)
        self.symptom_parser = JsonOutputParser(pydantic_object=SymptomResult)
        self.disease_parser = JsonOutputParser(pydantic_object=DiseaseResult)

    def _to_base64(self, image: Image.Image, max_side: int) -> str:
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _infer(self, image_b64: str, user_text: str, max_new_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            }],
            max_tokens=max_new_tokens,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def _safe_bbox(self, bbox: List[int]) -> List[int]:
        if bbox == [250, 250, 750, 750] or bbox == [0, 0, 0, 0]:
            return [0, 0, 0, 0]
        return [max(0, min(1000, v)) for v in bbox]

    def classify_crop(self, image: Image.Image, crops: List[str]) -> Dict:
        image_b64 = self._to_base64(image, max_side=448)
        prompt = (
            f"{SYSTEM}\nTask: What crop is in this image?\n"
            f"Valid: {', '.join(crops)}\n"
            f"Schema: {self.crop_parser.get_format_instructions()}"
        )
        raw = self._infer(image_b64, prompt, max_new_tokens=25)
        try:    return self.crop_parser.parse(raw)
        except: return {"crop": "unknown"}

    def detect_pest(self, image: Image.Image, crop: str, pests: List[str]) -> Dict:
        image_b64 = self._to_base64(image, max_side=1120)
        prompt = (
            f"{SYSTEM}\n"
            f"Task: Detect a LIVE INSECT on this {crop} plant.\n"
            f"Step 1 — Scan the entire image every inch carefully for any small dot like pests/insects, visula insect body, legs, or wings.\n"
            f"Step 2 — If found, note which quadrant it is in (top-left/top-right/bottom-left/bottom-right).\n"
            f"Step 3 — Output a tight bbox around the insect only.\n"
            f"Valid pests: {', '.join(pests)} select strictly from these pests only.\n"
            f"Schema: {self.pest_parser.get_format_instructions()}" 
        )
        raw = self._infer(image_b64, prompt, max_new_tokens=80)
        try:
            res = self.pest_parser.parse(raw)
            res["bbox"] = self._safe_bbox(res["bbox"])
            return res
        except:
            return {"name":"none","type":"pest","present":False,"confidence":0.0,"bbox":[0,0,0,0]}

    def detect_symptom(self, image: Image.Image, crop: str, pests: List[str]) -> Dict:
        image_b64 = self._to_base64(image, max_side=1120)
        prompt = (
            f"{SYSTEM}\n"
            f"Task: Detect PEST DAMAGE on this {crop} plant (holes, bite marks, rolled leaves, entry holes).\n"
            f"Step 1 — Look for irregular holes, torn edges, or frass deposits, color changes.\n"
            f"Step 2 — Note which part of the image the damage appears in.\n"
            f"Step 3 — Output a tight bbox around the most prominent damaged area only.\n"
            f"Valid pests causing damage: {', '.join(pests)} select strictly from these pests only.\n"
            f"Schema: {self.symptom_parser.get_format_instructions()}"
        )
        raw = self._infer(image_b64, prompt, max_new_tokens=80)
        try:
            res = self.symptom_parser.parse(raw)
            res["bbox"] = self._safe_bbox(res["bbox"])
            return res
        except:
            return {"name":"none","type":"symptom","present":False,"confidence":0.0,"bbox":[0,0,0,0]}

    def detect_disease(self, image: Image.Image, crop: str, diseases: List[str]) -> Dict:
        image_b64 = self._to_base64(image, max_side=1120)
        prompt = (
            f"{SYSTEM}\n"
            f"Task: Detect DISEASE on this {crop} plant (lesions, spots, mold, blight, discoloration).\n"
            f"Step 1 — Scan for abnormal coloration, necrotic spots, or fungal growth.\n"
            f"Step 2 — Note where the most severe symptom appears in the image.\n"
            f"Step 3 — Output a tight bbox around that region only.\n"
            f"Valid diseases: {', '.join(diseases)} select strictly from these diseases only.\n"
            f"Schema: {self.disease_parser.get_format_instructions()}"
        )
        raw = self._infer(image_b64, prompt, max_new_tokens=80)
        try:
            res = self.disease_parser.parse(raw)
            res["bbox"] = self._safe_bbox(res["bbox"])
            return res
        except:
            return {"name":"none","type":"disease","present":False,"confidence":0.0,"bbox":[0,0,0,0]}

    def fallback_bbox(self, image: Image.Image, crop: str, category: str, name: str) -> List[Dict]:
        w, h      = image.size
        image_b64 = self._to_base64(image, max_side=1120)

        instructions = {
            "pest":    f"Locate the LIVE INSECT ({name}) visible in this image.",
            "symptom": f"Locate the PEST DAMAGE ({name}) — holes, bite marks, rolled leaves.",
            "disease": f"Locate the DISEASE SYMPTOM ({name}) — spots, lesions, mold, discoloration.",
        }

        prompt = (
            f"You are an Agricultural AI.\n"
            f"{instructions.get(category, 'Locate the affected region.')}\n"
            f"Step 1 — Find where it appears in the image.\n"
            f"Step 2 — Note the quadrant: top-left / top-right / bottom-left / bottom-right.\n"
            f"Step 3 — Reply ONLY with JSON: {{\"ymin\": int, \"xmin\": int, \"ymax\": int, \"xmax\": int}}\n"
            f"Coordinates 0-1000. Top-left=(0,0), bottom-right=(1000,1000). Be precise."
        )

        try:
            raw   = self._infer(image_b64, prompt, max_new_tokens=40)
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if not match:
                return []
            coords = json.loads(match.group())
            ymin = max(0, min(1000, int(coords["ymin"])))
            xmin = max(0, min(1000, int(coords["xmin"])))
            ymax = max(0, min(1000, int(coords["ymax"])))
            xmax = max(0, min(1000, int(coords["xmax"])))

            if [ymin, xmin, ymax, xmax] in [[0,0,0,0], [250,250,750,750]]:
                return []

            bbox_abs = [
                int(xmin * w / 1000), int(ymin * h / 1000),
                int(xmax * w / 1000), int(ymax * h / 1000),
            ]
            print(f"  [Fallback bbox] '{name}' → {bbox_abs}")
            return [{"label": name, "confidence": 0.0, "bbox_abs": bbox_abs, "source": "fallback"}]

        except Exception as e:
            print(f"  [Fallback bbox] Failed for '{name}': {e}")
            return []


class YOLOLocalizationEngine:
    def __init__(self, model_repo: ModelRepository, conf_threshold: float = 0.10):
        self.repo           = model_repo
        self.conf_threshold = conf_threshold

    def localize(self, image: Image.Image, crop: str, category: str, qwen_label: str, conf_threshold: Optional[float] = None) -> List[Dict]:
        conf  = conf_threshold or self.conf_threshold
        model = self.repo.get_model(crop, category)

        if model is None:
            print(f"[YOLO] No model for {crop}/{category}")
            return []

        img_np  = np.array(image.convert("RGB"))
        results = model.predict(source=img_np, conf=conf, verbose=False)

        if not results or len(results[0].boxes) == 0:
            print(f"[YOLO] No boxes returned for {crop}/{category}")
            return []

        boxes = []
        for box in results[0].boxes:
            xyxy       = box.xyxy[0].tolist()
            confidence = round(float(box.conf.item()), 3)
            boxes.append({
                "label"     : qwen_label,
                "confidence": confidence,
                "bbox_abs"  : [int(v) for v in xyxy],
            })

        print(f"[YOLO] {crop}/{category} → {len(boxes)} box(es) for '{qwen_label}'")
        return boxes


class AgriGraphSystem:
    def __init__(self, vllm_host: str = VLLM_HOST):
        self.engine     = AgriInferenceEngine(vllm_host=vllm_host)
        self.kb         = AgriculturalKnowledgeBase()
        self.model_repo = ModelRepository()
        self.localizer  = YOLOLocalizationEngine(self.model_repo)

    def classifier_node(self, state: AgentState) -> Dict:
        t         = time.time()
        res       = self.engine.classify_crop(state["image_obj"], list(self.kb.crops.keys()))
        crop_name = res.get("crop", "").lower().strip()
        knowledge = self.kb.get_crop_knowledge(crop_name)
        print(f"[Classifier] {crop_name} | {time.time()-t:.2f}s")

        if knowledge:
            return {"crop_name": crop_name, "crop_knowledge": knowledge, "status": "analyzing", "encoded_inputs": None}
        return {"status": "error", "error": f"Crop '{crop_name}' not in knowledge base."}

    def parallel_detection_node(self, state: AgentState) -> Dict:
        if state.get("status") == "error":
            return {"detections": [], "yolo_detections": []}

        img    = state["image_obj"]
        crop   = state["crop_name"]
        kb     = state["crop_knowledge"]
        thresh = state["confidence_threshold"]

        tasks = {
            "pest":    lambda: self.engine.detect_pest(img, crop, kb.pests),
            "symptom": lambda: self.engine.detect_symptom(img, crop, kb.pests),
            "disease": lambda: self.engine.detect_disease(img, crop, kb.diseases),
        }

        results = {}
        t = time.time()

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fn): cat for cat, fn in tasks.items()}
            for future in as_completed(futures):
                cat = futures[future]
                try:
                    res = future.result()
                except Exception:
                    res = {"name":"none","type":cat,"present":False,"confidence":0.0,"bbox":[0,0,0,0]}
                res["category"] = cat
                results[cat] = res
                print(f"  [{cat:8s}] {res['name']} | present={res['present']} | conf={res['confidence']:.2f}")

        print(f"[Parallel detection] {time.time()-t:.2f}s")

        confirmed = [
            r for r in results.values()
            if r["present"] and r["confidence"] >= thresh
        ]
        return {"detections": confirmed, "yolo_detections": []}

    def yolo_localization_node(self, state: AgentState) -> Dict:
        if state.get("status") == "error" or not state.get("detections"):
            return {"yolo_detections": []}

        img        = state["image_obj"]
        crop       = state["crop_name"]
        detections = state["detections"]

        def run_yolo(det: Dict) -> List[Dict]:
            category   = det["category"]
            qwen_label = det["name"]
            t          = time.time()

            boxes = self.localizer.localize(img, crop, category, qwen_label)
            print(f"  [YOLO/{category}] '{qwen_label}' → {len(boxes)} box(es) | {time.time()-t:.2f}s")

            if not boxes:
                print(f"  [YOLO/{category}] No boxes — fallback bbox agent for '{qwen_label}'")
                boxes = self.engine.fallback_bbox(img, crop, category, qwen_label)

            for b in boxes:
                b["category"]  = category
                b["qwen_name"] = qwen_label

            return boxes

        all_yolo_boxes = []
        t = time.time()

        with ThreadPoolExecutor(max_workers=max(1, len(detections))) as pool:
            futures = [pool.submit(run_yolo, det) for det in detections]
            for future in as_completed(futures):
                try:
                    all_yolo_boxes.extend(future.result())
                except Exception as e:
                    print(f"  [YOLO] Error: {e}")

        print(f"[YOLO localization] {time.time()-t:.2f}s | total boxes: {len(all_yolo_boxes)}")
        return {"yolo_detections": all_yolo_boxes}


def build_app_graph(system: AgriGraphSystem):
    workflow = StateGraph(AgentState)
    workflow.add_node("classifier",         system.classifier_node)
    workflow.add_node("parallel_detection", system.parallel_detection_node)
    workflow.add_node("yolo_localization",  system.yolo_localization_node)

    workflow.set_entry_point("classifier")
    workflow.add_edge("classifier",         "parallel_detection")
    workflow.add_edge("parallel_detection", "yolo_localization")
    workflow.add_edge("yolo_localization",  END)

    return workflow.compile()


@st.cache_resource
def load_agri_logic():
    sys   = AgriGraphSystem(vllm_host=VLLM_HOST)
    graph = build_app_graph(sys)
    return graph


def optimize_image(img: Image.Image, max_size_mb: float = 1.0, max_dim: int = 1280) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    size_mb = buffer.tell() / (1024 * 1024)
    if size_mb <= max_size_mb:
        return img
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def draw_yolo_bboxes(image: Image.Image, yolo_detections: List[Dict]) -> Image.Image:
    canvas = image.copy()
    draw   = ImageDraw.Draw(canvas)
    colors = {"pest": "red", "symptom": "orange", "disease": "purple"}

    for det in yolo_detections:
        xmin, ymin, xmax, ymax = det["bbox_abs"]
        color       = colors.get(det.get("category", "pest"), "red")
        is_fallback = det.get("source") == "fallback"
        label       = f"{det['qwen_name']} {det['confidence']:.2f}" + (" [VLM]" if is_fallback else "")
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3 if is_fallback else 4)
        draw.text((xmin, max(ymin - 18, 0)), label, fill=color)

    return canvas


def run_crop_doctor():
    st.set_page_config(page_title="CropDoctor AI", layout="centered")
    graph_app = load_agri_logic()

    st.markdown(
        "<h1 style='text-align:center;'>CropDoctor AI</h1>"
        "<p style='text-align:center;color:gray;'>Intelligent Crop Health Analysis</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload Crop Image", type=["jpg","jpeg","png"])
    if not uploaded:
        st.info("Please upload a crop image to begin analysis.")
        return

    img = Image.open(uploaded).convert("RGB")
    img = optimize_image(img)
    st.image(img, use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing crop health..."):
            t0 = time.time()
            output = graph_app.invoke({
                "image_obj": img, "crop_name": "", "crop_knowledge": None,
                "detections": [], "yolo_detections": [], "confidence_threshold": 0.4,
                "status": "start", "error": None, "encoded_inputs": None,
            })
            elapsed = round(time.time() - t0, 2)

        if output.get("error"):
            st.error(output["error"])
            return

        crop_name       = output.get("crop_name", "Unknown")
        detections      = output.get("detections", [])
        yolo_detections = output.get("yolo_detections", [])

        pest_names    = [d["name"] for d in detections if d.get("category") == "pest"]
        symptom_names = [d["name"] for d in detections if d.get("category") == "symptom"]
        disease_names = [d["name"] for d in detections if d.get("category") == "disease"]

        st.markdown("---")
        st.subheader("Analysis Report")
        st.markdown(f"Crop: **{crop_name.upper()}**")
        st.markdown(f"Pest: **{', '.join(pest_names) if pest_names else 'Clear'}**")
        st.markdown(f"Pest Damage: **{', '.join(symptom_names) if symptom_names else 'Clear'}**")
        st.markdown(f"Disease: **{', '.join(disease_names) if disease_names else 'Clear'}**")
        st.caption(f"Processing Time: {elapsed}s")

        st.markdown("---")
        st.subheader("Detected Regions")
        cols = st.columns(3)

        qwen_by_cat = {
            "pest":    pest_names,
            "symptom": symptom_names,
            "disease": disease_names,
        }

        for col, (cat, title) in zip(cols, [("pest","🪲 Pests"),("symptom","🔍 Damage"),("disease","🤒 Disease")]):
            with col:
                st.markdown(f"**{title}**")
                cat_boxes  = [d for d in yolo_detections if d.get("category") == cat]
                qwen_names = qwen_by_cat[cat]

                if cat_boxes:
                    annotated = draw_yolo_bboxes(img, cat_boxes)
                    st.image(annotated, use_container_width=True)
                    for b in cat_boxes:
                        src = " (VLM fallback)" if b.get("source") == "fallback" else ""
                        st.caption(f"{b['qwen_name']} | conf: {b['confidence']:.2f}{src}")
                elif qwen_names:
                    st.image(img, use_container_width=True)
                    st.warning(f"**{', '.join(qwen_names)}** detected — bbox unavailable")
                else:
                    st.image(img, use_container_width=True)
                    st.caption("Clear ✅")


if __name__ == "__main__":
    run_crop_doctor()
