"""Real-time ASL / FSL recognition baseline (extends the instructor starter).

    python main.py [--lang ASL] [--camera 0]

Keys
    A / F       switch ASL / FSL (loads that language's own models, resets state)
    S           speak transcript (language-aware voice if installed)
    E           toggle transcript EDIT mode (type text, Backspace deletes a char, Enter/Esc exits)
    Backspace   delete last word (normal mode)      C  clear transcript
    B           measure latency/FPS over the next 120 frames -> 06_Results/latency_fps.csv
    Q / Esc     quit
Left hand is drawn orange, right hand green. This is a limited-vocabulary baseline,
not a translator.
"""
import argparse
import textwrap
import time

import cv2
import mediapipe as mp

from inference import (HandTracker, LatencyMeter, SignRecognizer, SignStabilizer, Speaker)
from preprocessing import assign_slots, MODELS_DIR

BENCH_FRAMES = 120
COLORS = {0: (0, 140, 255), 1: (0, 200, 0)}   # BGR: Left orange, Right green
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils


def put(frame, text, org, scale=0.65, color=(0, 255, 255), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ASL", choices=["ASL", "FSL"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check permissions and camera index.")
    tracker, speaker = HandTracker(), Speaker()
    recognizers = {}
    lang = a.lang
    stab = SignStabilizer()
    transcript, edit_mode = "", False
    meter, bench = LatencyMeter(), None
    last_loop = time.perf_counter()
    toast, toast_until = "", 0.0

    def get_rec(l):
        if l not in recognizers:
            recognizers[l] = SignRecognizer(l, a.models_dir)
        return recognizers[l]

    try:
        while True:
            ok, frame_in = cap.read()
            if not ok:
                print("Camera frame unavailable; stopping.")
                break
            t0 = time.perf_counter()
            frame, res, raw, mask = tracker.process(frame_in)
            rec = get_rec(lang)
            pred = rec.update(raw, mask, t0)
            out = stab.step(pred, t0)
            lat = time.perf_counter() - t0
            if out["emit"]:
                transcript += out["emit"] + " "
            if bench is not None:
                bench.record(lat, t0)
                if bench.summary()["frames"] >= BENCH_FRAMES:
                    row = bench.write_csv(lang)
                    toast = f"Saved: {row['avg_latency_ms']:.1f} ms avg, {row['measured_fps']:.1f} FPS"
                    toast_until, bench = time.perf_counter() + 4, None
            now = time.perf_counter()
            fps = 1.0 / max(now - last_loop, 1e-6)
            last_loop = now

            for slot, lm in assign_slots(res):
                spec = mp_draw.DrawingSpec(color=COLORS[slot], thickness=2, circle_radius=3)
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS, spec, spec)
            side = {(True, False): "LEFT", (False, True): "RIGHT", (True, True): "BOTH"}.get(
                (bool(mask[0]), bool(mask[1])), "NONE")

            put(frame, f"Mode: {lang} | Hands: {side} | FPS: {fps:.1f} | {lat * 1000:.0f} ms", (15, 30))
            label = out["label"]
            col = (0, 0, 255) if label == "UNKNOWN" else (0, 255, 0)
            put(frame, f"{label}  {out['conf'] * 100:.0f}%  [{out['state']}/{pred.source}]",
                (15, 65), 0.8, col)
            for i, msg in enumerate(rec.messages):
                put(frame, msg[:70], (15, 95 + 26 * i), 0.55, (0, 0, 255))
            if bench is not None:
                put(frame, f"Benchmarking {bench.summary()['frames']}/{BENCH_FRAMES}", (15, 125), 0.55)
            if time.perf_counter() < toast_until:
                put(frame, toast, (15, 150), 0.6, (255, 255, 0))
            h = frame.shape[0]
            lines = textwrap.wrap(transcript + ("_" if edit_mode else ""), 55) or [""]
            for i, ln in enumerate(lines[-3:]):
                put(frame, ln, (15, h - 75 + 26 * i), 0.7, (255, 255, 255))
            put(frame, "EDIT MODE (Enter=done)" if edit_mode else
                "A/F lang  S speak  E edit  Bksp del word  C clear  B bench  Q quit",
                (15, h - 12), 0.5, (200, 200, 200), 1)
            cv2.imshow("ASL / FSL Recognition Baseline", frame)

            k = cv2.waitKey(1)
            if k == -1:
                continue
            k &= 0xFF
            if edit_mode:
                if k in (13, 10, 27):
                    edit_mode = False
                elif k in (8, 127):
                    transcript = transcript[:-1]
                elif 32 <= k <= 126:
                    transcript += chr(k)
                continue
            if k in (ord("q"), 27):
                break
            elif k in (ord("a"), ord("f")):
                new = "ASL" if k == ord("a") else "FSL"
                if new != lang:
                    lang = new
                    get_rec(lang).reset(); stab.reset()
            elif k == ord("e"):
                edit_mode = True
            elif k in (8, 127):
                transcript = transcript.rstrip().rsplit(" ", 1)[0] + " " if " " in transcript.rstrip() else ""
            elif k == ord("c"):
                transcript = ""
            elif k == ord("s"):
                if speaker.ok:
                    _, exact = speaker.voice_for(lang)
                    speaker.say(transcript, lang)
                    toast = "Speaking" + ("" if exact else f" (no {lang}-specific voice found, default voice)")
                else:
                    toast = "TTS unavailable (pip install pyttsx3)"
                toast_until = time.perf_counter() + 3
            elif k == ord("b"):
                bench = LatencyMeter()
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
