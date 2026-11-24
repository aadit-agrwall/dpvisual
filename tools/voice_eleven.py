import os, sys, json, hashlib, subprocess, pathlib, shutil
import requests
ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "media/voice/eleven_cache"
CHOICE = ROOT / "media/voice/eleven_choice.json"
API = "https://api.elevenlabs.io/v1"
SR = os.environ.get("ELEVEN_SR", "48000")
CACHE.mkdir(parents=True, exist_ok=True)

def key():
    k = os.environ.get("ELEVEN_API_KEY")
    envf = ROOT / ".env"
    if not k and envf.exists():
        for ln in envf.read_text().splitlines():
            if ln.startswith("ELEVEN_API_KEY="):
                k = ln.split("=", 1)[1].strip().strip('"')
    if not k:
        sys.exit("no ELEVEN_API_KEY (run the key step)")
    return k

def hdr():
    return {"xi-api-key": key()}

def choice():
    if CHOICE.exists():
        return json.loads(CHOICE.read_text())
    return {"voice_id": None, "name": None, "model": "eleven_multilingual_v2",
            "stability": 0.45, "similarity": 0.8, "style": 0.25}

def synth(text, vid, model, vs, mp3_path):
    r = requests.post(f"{API}/text-to-speech/{vid}?output_format=mp3_44100_128", headers=hdr(),
                      json={"text": text, "model_id": model, "voice_settings": vs}, timeout=180)
    if r.status_code != 200:
        sys.exit(f"ElevenLabs error {r.status_code}: {r.text[:400]}")
    pathlib.Path(mp3_path).write_bytes(r.content)

def say(text, voice_id=None, out=None):
    ch = choice()
    vid = voice_id or ch["voice_id"]
    if not vid:
        sys.exit("no voice chosen yet (use: voice_eleven.py use VOICE_ID NAME)")
    model = os.environ.get("ELEVEN_MODEL", ch.get("model", "eleven_multilingual_v2"))
    vs = {"stability": ch.get("stability", 0.45), "similarity_boost": ch.get("similarity", 0.8),
          "style": ch.get("style", 0.25), "use_speaker_boost": True}
    sp = float(os.environ.get("ELEVEN_SPEED", "1.0"))
    if sp != 1.0:
        vs["speed"] = sp
    kk = lambda sr: hashlib.sha1(json.dumps([vid, model, vs, text, sr], sort_keys=True).encode()).hexdigest()[:16]
    wav = CACHE / (kk(SR) + '.wav')
    if not wav.exists():
        mp3 = CACHE / (kk('24000') + '.mp3')
        if not mp3.exists():
            synth(text, vid, model, vs, mp3)
        subprocess.run(['ffmpeg', '-nostdin', '-y', '-loglevel', 'error', '-i', str(mp3), '-ar', SR, '-ac', '1', str(wav)], check=True)
    if out:
        shutil.copy(wav, out)
    return str(wav)

def male_voices():
    r = requests.get(f"{API}/voices", headers=hdr(), timeout=60)
    if r.status_code != 200:
        sys.exit(f"voices error {r.status_code}: {r.text[:300]}")
    vs = r.json().get("voices", [])
    m = [v for v in vs if (v.get("labels") or {}).get("gender", "").lower() == "male"]
    return m, vs

def info():
    r = requests.get(f"{API}/user/subscription", headers=hdr(), timeout=60)
    if r.status_code != 200:
        sys.exit(f"key check failed {r.status_code}: {r.text[:300]}")
    d = r.json()
    print("plan:", d.get("tier"), "| characters used:", d.get("character_count"), "of", d.get("character_limit"))

EN = "Every face here is one point in pixel space. Two hidden parameters, the head angles, generate them all. That surface is a manifold."
HI = "यह हर face pixel space में एक point है। दो hidden parameters, यानी head angles, इन सब को generate करते हैं। इस surface को कहते हैं manifold।"

def audition(limit=6):
    m, allv = male_voices()
    print(f"{len(allv)} voices in account, {len(m)} male")
    ch = choice()
    vs = {"stability": 0.45, "similarity_boost": 0.8, "style": 0.25, "use_speaker_boost": True}
    lines = []
    for i, v in enumerate(m[:limit], 1):
        lab = v.get("labels") or {}
        name = "".join(c for c in v["name"] if c.isalnum())
        lines.append(f"{i:02d} {v['name']} | {v['voice_id']} | {lab.get('accent','')} {lab.get('description','')} {lab.get('age','')}")
        for tag, txt in (("en", EN), ("hinglish", HI)):
            f = ROOT / f"media/voice/audition_eleven/{i:02d}_{name}_{tag}.mp3"
            if not f.exists():
                print("making", i, v["name"], tag, flush=True)
                synth(txt, v["voice_id"], "eleven_multilingual_v2", vs, f)
    (ROOT / "media/voice/audition_eleven/names.txt").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "info": info()
    elif c == "audition": audition(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif c == "use":
        ch = choice(); ch["voice_id"] = sys.argv[2]; ch["name"] = sys.argv[3] if len(sys.argv) > 3 else ""
        CHOICE.parent.mkdir(parents=True, exist_ok=True); CHOICE.write_text(json.dumps(ch, indent=1)); print("saved", ch)
    elif c == "say": print(say(sys.argv[2], out=sys.argv[3] if len(sys.argv) > 3 else None))
    else: print("commands: info | audition [N] | use VOICE_ID [NAME] | say \"text\" [out.wav]")
