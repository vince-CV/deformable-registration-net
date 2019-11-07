import tensorflow as tf
from models import DIRNet
from config import get_config
from data import AryllaDataHandler
from ops import mkdir

def main():
  sess = tf.Session()
  config = get_config(is_train=False)
  mkdir(config.result_dir)

  reg = DIRNet(sess, config, "DIRNet", is_train=False)
  reg.restore(config.ckpt_dir)
  dh = AryllaDataHandler("vince_data",  is_train=False)

  for i in range(2):
    result_i_dir = config.result_dir+"/{}".format(i)
    mkdir(result_i_dir)

    batch_x, batch_y = dh.sample_pair(config.batch_size, i)
    reg.deploy(result_i_dir, batch_x, batch_y)

if __name__ == "__main__":
  main()
