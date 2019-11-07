class Config(object):
  pass

def get_config(is_train):
  
  config = Config()

  if is_train:
    config.batch_size = 32
    config.im_size = [40, 40]
    config.lr = 1e-4
    config.iteration = 50000
    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
    config.result_dir = "result"

  else:
    config.batch_size = 30
    config.im_size = [40, 40]
    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
    
  return config
