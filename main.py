import os
import sys
import importlib

# --- Khởi tạo PATH an toàn (hỗ trợ Google Drive Shortcut/Symlink) ---
# Dùng realpath để theo dõi symlink thực sự trên FUSE Drive
_script = os.path.realpath(__file__)
BASE_DIR = os.path.dirname(_script)
# Cũng thêm abspath phòng trường hợp realpath và abspath khác nhau
_alt = os.path.dirname(os.path.abspath(__file__))

for _d in [BASE_DIR, _alt]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.chdir(BASE_DIR)

# Xóa cache module cũ của vphoberttagger (phòng import nhầm namespace pkg)
for _k in list(sys.modules.keys()):
    if _k == 'vphoberttagger' or _k.startswith('vphoberttagger.'):
        del sys.modules[_k]
importlib.invalidate_caches()

# Kiểm tra __init__.py tồn tại vật lý trước khi import
_init = os.path.join(BASE_DIR, 'vphoberttagger', '__init__.py')
if not os.path.isfile(_init):
    raise RuntimeError(
        f"KHÔNG TÌM THẤY vphoberttagger/__init__.py tại '{BASE_DIR}'!\n"
        f"Hãy chạy lại ô Bước 2.1 trong notebook để đồng bộ mã nguồn.\n"
        f"Các file hiện có: {os.listdir(os.path.join(BASE_DIR, 'vphoberttagger'))}"
    )

from vphoberttagger import LOGGER, Trainer, Predictor

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        LOGGER.info("Start TRAIN process...")
        Trainer.train()
    elif sys.argv[1] == 'test':
        LOGGER.info("Start TEST process...")
        Trainer.test()
    elif sys.argv[1] == 'predict':
        LOGGER.info("Start PREDICT process...")
        Predictor.tagging()
    elif sys.argv[1] == 'demo':
        LOGGER.info("Start PREDICT process...")
        comd = "PYTHONPATH=./ streamlit run vphoberttagger/demo.py -- " + " ".join(sys.argv[1:])
        print(comd)
        os.system(comd)

else:
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` not found!!!')