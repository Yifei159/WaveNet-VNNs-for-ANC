import os
import json
import toml
import torch
import argparse
import numpy as np
import sounddevice as sd
from threading import Thread, Event
from scipy.io import loadmat

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    from scipy.io import wavfile
    _HAS_SF = False

from networks import WaveNet_VNNs
from utils import fir_filter, SEF

def _to_b1t(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"_to_b1t: unsupported shape {tuple(x.shape)}")

def _to_t(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.squeeze(1).squeeze(0)
    if x.dim() == 2:
        return x.squeeze(0)
    if x.dim() == 1:
        return x
    raise ValueError(f"_to_t: unsupported shape {tuple(x.shape)}")

def load_model_and_channels(ckpt_path: str, device: torch.device):
    base_dir = os.path.dirname(os.path.abspath(ckpt_path))

    cfg_json = os.path.join(base_dir, 'config.json')
    if not os.path.isfile(cfg_json):
        raise FileNotFoundError(f'缺少 config.json: {cfg_json}')
    with open(cfg_json, 'r') as f:
        net_config = json.load(f)

    cfg_toml = os.path.join(base_dir, 'cfg_train.toml')
    if not os.path.isfile(cfg_toml):
        raise FileNotFoundError(f'缺少 cfg_train.toml: {cfg_toml}')
    cfg = toml.load(cfg_toml)
    eta = float(cfg.get('trainer', {}).get('eta', 0.1))
    sr = int(cfg.get('listener', {}).get('listener_sr', 16000))

    model = WaveNet_VNNs(net_config)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    if any(k.startswith('module.') for k in state.keys()):
        from collections import OrderedDict
        state = OrderedDict((k.replace('module.', ''), v) for k, v in state.items())

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    pri_mat = os.path.join(base_dir, 'pri_channel.mat')
    sec_mat = os.path.join(base_dir, 'sec_channel.mat')
    if not os.path.isfile(pri_mat):
        raise FileNotFoundError(f'缺少 pri_channel.mat: {pri_mat}')
    if not os.path.isfile(sec_mat):
        raise FileNotFoundError(f'缺少 sec_channel.mat: {sec_mat}')

    Pri = torch.tensor(loadmat(pri_mat, mat_dtype=True)['pri_channel']).squeeze().to(device=device, dtype=torch.float32)
    Sec = torch.tensor(loadmat(sec_mat, mat_dtype=True)['sec_channel']).squeeze().to(device=device, dtype=torch.float32)

    return model, Pri, Sec, eta, sr

class Toggle:
    def __init__(self):
        self.on = True
        self.stop = Event()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt',
        default=os.path.join('trained_model', 'WaveNetVNNs_nonlinear0.5.pth'),
    )
    parser.add_argument('--blocksize', type=int, default=512, help='lower for better ANC, but higher computational comlexity')
    parser.add_argument('--device_out', type=int, default=None)
    parser.add_argument('--dtype', default='float32', choices=['float32', 'int16'])
    parser.add_argument('--wav', default=os.path.join('test_dataset', 'babble-16000nom.wav'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'no model: {ckpt_path}')

    model, Pri, Sec, eta, sr = load_model_and_channels(ckpt_path, device)

    wav_path = args.wav
    if not os.path.isabs(wav_path):
        wav_path = os.path.join(os.path.dirname(__file__), wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f'no test audio: {wav_path}')

    if _HAS_SF:
        ref_all, file_sr = sf.read(wav_path, dtype='float32', always_2d=True)
        ref_all = ref_all[:, 0]
    else:
        file_sr, data = wavfile.read(wav_path)
        if data.dtype == np.int16:
            ref_all = (data.astype(np.float32) / 32768.0)
        elif data.dtype == np.int32:
            ref_all = (data.astype(np.float32) / 2147483648.0)
        elif data.dtype == np.float32:
            ref_all = data.astype(np.float32)
        else:
            ref_all = data.astype(np.float32)
        if ref_all.ndim == 2:
            ref_all = ref_all[:, 0]
    if file_sr != sr:
        raise ValueError(f'sampling rate mis match')

    torch_dtype = torch.float32
    with torch.no_grad():
        dummy = torch.zeros(1, 1, args.blocksize, device=device, dtype=torch_dtype)
        y_dummy = model(dummy)
        print('[SELFTEST]', 'input', tuple(dummy.shape), '-> output', tuple(y_dummy.shape))

    sd_dtype = args.dtype
    np_dtype = np.float32 if sd_dtype == 'float32' else np.int16

    toggle = Toggle()

    def key_listener():
        try:
            while not toggle.stop.is_set():
                s = input().strip().lower()
                if s == 'm':
                    toggle.on = not toggle.on
                    print(f"ANC on" if toggle.on else "ANC off")
                elif s == 'q':
                    toggle.stop.set()
        except EOFError:
            toggle.stop.set()

    Thread(target=key_listener, daemon=True).start()

    idx = 0
    total = len(ref_all)

    try:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            print(f'  #{i}: {d["name"]} (in:{d["max_input_channels"]}, out:{d["max_output_channels"]})')
    except Exception as e:
        print(f'no speaker: {e}')

    def audio_callback(outdata, frames, time_info, status):
        nonlocal idx
        try:
            if status:
                print(status)

            if toggle.stop.is_set():
                outdata[:] = 0
                return

            end = min(idx + frames, total)
            ref_blk = ref_all[idx:end]
            N = ref_blk.shape[0]

            if N < frames:
                pad = np.zeros(frames - N, dtype=np.float32)
                ref_blk = np.concatenate([ref_blk.astype(np.float32, copy=False), pad], axis=0)
            else:
                ref_blk = ref_blk.astype(np.float32, copy=False)

            if toggle.on:
                ref_t = torch.from_numpy(ref_blk).to(device=device, dtype=torch_dtype)
                ref_b1t = _to_b1t(ref_t)

                with torch.no_grad():
                    y = model(ref_b1t)
                    y_t = _to_t(y)
                    y_nl_t = SEF(y_t, eta)
                    dn_b1t = fir_filter(Sec, _to_b1t(y_nl_t))
                    target_b1t = fir_filter(Pri, ref_b1t)
                    en_b1t = dn_b1t + target_b1t
                    out_t = _to_t(en_b1t)

                out_blk = out_t.detach().float().clamp(-1.0, 1.0).cpu().numpy()
            else:
                out_blk = ref_blk

            stereo = np.stack([out_blk, out_blk], axis=1)

            if sd_dtype == 'int16':
                stereo_i16 = np.empty_like(stereo, dtype=np.int16)
                stereo_i16[:, 0] = np.clip(stereo[:, 0] * 32767.0, -32768, 32767).astype(np.int16)
                stereo_i16[:, 1] = np.clip(stereo[:, 1] * 32767.0, -32768, 32767).astype(np.int16)
                outdata[:] = stereo_i16
            else:
                outdata[:] = stereo.astype(np_dtype, copy=False)

            idx = end
            if idx >= total:
                toggle.stop.set()

        except Exception as e:
            print('[ERROR][callback]', repr(e))
            outdata[:] = 0
            toggle.stop.set()

    stream = sd.OutputStream(
        samplerate=sr,
        blocksize=args.blocksize,
        dtype=sd_dtype,
        channels=2,
        callback=audio_callback,
        device=args.device_out if (args.device_out is not None) else None,
        latency='low'
    )

    try:
        with stream:
            while not toggle.stop.is_set():
                sd.sleep(100)
    except KeyboardInterrupt:
        pass
    finally:
        toggle.stop.set()

if __name__ == '__main__':
    main()