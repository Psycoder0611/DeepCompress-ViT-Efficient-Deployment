"""Run from the repo root:  python ours/smoke_train.py --rank 16 --rank_proj 8
CPU smoke test of the patched main.py: random weights, fake 2-batch loader, tiny ranks.
Checks per-type ranks flow through encoder/decoder/loss/backward/checkpoint. NOT a training run."""
import sys, os, runpy, torch, timm
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in ours/)
os.chdir(REPO); sys.path.insert(0, REPO)
extra = sys.argv[1:]   # e.g. --rank 16 --rank_proj 8   or   --rank 16 --time_iters 2
# --- CPU shims (the authors' code calls .cuda() everywhere) ---
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.empty_cache = lambda *a, **k: None
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
_create = timm.create_model
timm.create_model = lambda name, pretrained=True, **kw: _create(name, pretrained=False, **kw)
_load = torch.load
def fake_load(path, *a, **k):
    if str(path).endswith('deit_small_cifar10_best.pth'):
        return _create('deit_small_patch16_224', pretrained=False, num_classes=10).state_dict()
    return _load(path, *a, **k)
torch.load = fake_load
import src.utils.data_utils as du, src.utils as su
torch.manual_seed(0)
def fake_loaders(bs=4, root=None, num_workers=0):
    data = [(torch.randn(bs, 3, 224, 224), torch.randint(0, 10, (bs,))) for _ in range(2)]
    return data, data
du.get_dataloaders_cifar10 = fake_loaders; su.get_dataloaders_cifar10 = fake_loaders
sys.argv = ['main.py', '--dataset', 'cifar10', '--device', 'cpu', '--batch_size', '4', '--epochs', '1',
            '--initial_iters', '2', '--base_dir', 'smoke', '--eval_interval', '200000'] + extra
runpy.run_path('main.py', run_name='__main__')
