import glob
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps

from vision.predict import explain, load_checkpoint

st.set_page_config(page_title="Image Classifier + Grad-CAM", page_icon="🌿", layout="wide")

if not os.path.exists("models/best.pt"):
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def get_bundle():
    return load_checkpoint("models/best.pt")


bundle = get_bundle()
metrics = json.load(open("models/metrics.json")) if os.path.exists("models/metrics.json") else None

st.title("🌿 Image Classifier with Grad-CAM")
st.caption(f"{bundle.arch} fine-tuned on {len(bundle.classes)} classes · running on {bundle.device}")
alpha = st.sidebar.slider("Heatmap strength", 0.1, 0.9, 0.45, 0.05)
st.sidebar.markdown("**Supported classes**")
st.sidebar.caption(", ".join(sorted(bundle.classes)))

tab_cls, tab_rep = st.tabs(["📷 Classify", "📊 Model report"])

# ------------------------------ classify ------------------------------
with tab_cls:
    samples = {os.path.basename(p).split("__", 1)[1][:-4].replace("_", " ") + f" (#{i + 1})": p
               for i, p in enumerate(sorted(glob.glob("models/samples/*.jpg")))}
    left, right = st.columns(2)
    up = left.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
    pick = right.selectbox("…or try a held-out test image (true label shown in the name)", ["-"] + list(samples))
    image = None
    if up is not None:
        image = ImageOps.exif_transpose(Image.open(up))
    elif pick != "-":
        image = Image.open(samples[pick])

    if image is None:
        st.info("Upload a photo of one of the supported classes to see the prediction and what the model looked at.")
    else:
        base = explain(bundle, image, None, alpha=alpha)
        options = [f"{n} ({p:.0%})" for n, p, _ in base["top"]]
        choice = st.selectbox("Explain which class?", range(len(options)), format_func=lambda i: options[i])
        res = base if choice == 0 else explain(bundle, image, base["top"][choice][2], alpha=alpha)

        top_name, top_p, _ = base["top"][0]
        if top_p < 0.5:
            st.warning(f"Low confidence ({top_p:.0%}). The photo may show something outside the supported classes - "
                       "a softmax classifier always has to pick *something*.")
        else:
            st.success(f"Prediction: **{top_name}** ({top_p:.0%})")
        c1, c2, c3 = st.columns([1, 1, 1.1])
        c1.image(res["crop"], caption="What the model sees (center crop)", width="stretch")
        c2.image(res["overlay"], caption=f"Grad-CAM for: {bundle.classes[res['explained']]}", width="stretch")
        df = pd.DataFrame(base["top"], columns=["class", "probability", "idx"])
        fig = px.bar(df.iloc[::-1], x="probability", y="class", orientation="h", range_x=[0, 1], text_auto=".0%")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title="Top predictions", yaxis_title=None)
        c3.plotly_chart(fig, width="stretch")
        st.caption("Red/yellow = regions that pushed the model toward this class. It is a coarse visual hint, "
                   "not proof of how the model reasons.")

# ------------------------------ report ------------------------------
with tab_rep:
    if not metrics:
        st.info("No metrics.json found - run train.py to generate the report.")
    else:
        k = st.columns(4)
        k[0].metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        k[1].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        k[2].metric(f"Top-{metrics['k']} accuracy", f"{metrics['topk_accuracy']:.1%}")
        k[3].metric("Test images", f"{metrics['n_test']:,}")
        st.caption(f"{metrics['arch']} · {metrics['dataset']} · pretrained={metrics['pretrained']} · "
                   f"{metrics['epochs']} epochs · {metrics['train_seconds']}s on {metrics['device']}")
        if os.path.exists("models/history.csv"):
            h = pd.read_csv("models/history.csv")
            a, b = st.columns(2)
            a.plotly_chart(px.line(h, x="epoch", y=["train_loss", "val_loss"], markers=True, title="Loss"), width="stretch")
            b.plotly_chart(px.line(h, x="epoch", y=["train_acc", "val_acc"], markers=True, title="Accuracy"), width="stretch")
        pc = pd.DataFrame(metrics["per_class"]).T.reset_index(names="class").sort_values("f1")
        a, b = st.columns(2)
        a.plotly_chart(px.bar(pc.head(15), x="f1", y="class", orientation="h", range_x=[0, 1],
                              title="15 hardest classes (F1)"), width="stretch")
        b.markdown("**Most common confusions**")
        b.dataframe(pd.DataFrame(metrics["top_confusions"]), hide_index=True, width="stretch")
        st.caption("Test split was never used for training or model selection (the best epoch was chosen on validation data).")
