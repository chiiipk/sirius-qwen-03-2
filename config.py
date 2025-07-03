import os
from configparser import ConfigParser

class Config(ConfigParser):
    def __init__(self, config_file):
        print(f"----- Đang đọc cấu hình từ: {os.path.abspath(config_file)} -----")
        
        raw_config = ConfigParser()
        read_ok = raw_config.read(config_file, encoding='utf-8')
        
        # if not read_ok:
        #     print(f"\n!!! LỖI:  '{config_file}'.\n")
        #     raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_file}")

        # if not raw_config.sections():
        #     print(f"\n!!! LỖI: File cấu hình '{config_file}' bị trống hoặc không hợp lệ.\n")
        #     raise ValueError(f"File cấu hình không hợp lệ: {config_file}")
            
        print("----- Đọc cấu hình thành công. Đang xử lý giá trị... -----")
        self.cast_values(raw_config)
        print("----- Hoàn tất cấu hình. -----\n")

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                if isinstance(value, str):
                    value = value.strip()

                if value.startswith("[") and value.endswith("]"):
                    try:
                        val = eval(value)
                    except (SyntaxError, NameError):
                        val = value
                else:
                    for attr in ["getint", "getfloat", "getboolean"]:
                        try:
                            val = getattr(raw_config[section], attr)(key)
                            break
                        except (ValueError, KeyError):
                            val = value
                
                setattr(self, key, val)
