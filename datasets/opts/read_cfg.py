import yaml, os


class DictToClass(object):
    '''
    ;;将字典准换为 class 类型
    '''
    @classmethod
    def _to_class(cls, _obj):
        _obj_ = type('new', (object,), _obj)
        [setattr(_obj_, key, cls._to_class(value)) if isinstance(value, dict) else setattr(_obj_, key, value) for
         key, value in _obj.items()]
        return _obj_


class ReadConfigFiles(object):
    def __init__(self,yml_path):
        '''
        ;;获取当前工作路径
        '''
        self.yml_path = yml_path
 
    @classmethod
    def open_file(self):
        '''
        ;;读取当前工作目录下cfg.yaml文件内容，并返回字典类型
        :return:
        '''
        return yaml.load(
            open(self.yml_path, 'r', encoding='utf-8').read(), Loader=yaml.FullLoader
        )
 
    @classmethod
    def cfg(cls, item=None):
        '''
        ;;调用该方法获取需要的配置，item如果为None，返回则是全部配置
        :param item:
        :return:
        '''
        return DictToClass._to_class(cls.open_file().get(item) if item else cls.open_file())
