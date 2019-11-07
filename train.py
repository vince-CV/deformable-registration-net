import tensorflow as tf
import numpy as np
from models import DIRNet
from models import ResNet
from config import get_config
from data import AryllaDataHandler
from ops import mkdir

def main():
  sess = tf.Session()
  config = get_config(is_train=True)
  mkdir(config.tmp_dir)
  mkdir(config.ckpt_dir)
  mkdir(config.result_dir)

  DIRNet_Model = DIRNet(sess, config, "DIRNet_tr", is_train=True)

  Train_Data = AryllaDataHandler("Data_1",  is_train = True)
  Eval_Data =  AryllaDataHandler("Data_2",  is_train = False)

  txt_dir = config.result_dir

  for i in range(config.iteration):
    batch_x, batch_y = Train_Data.sample_pair(config.batch_size)
    train_loss = DIRNet_Model.fit(batch_x, batch_y)

    batch_x_eval, batch_y_eval = Eval_Data.sample_pair(config.batch_size)
    eval_loss = DIRNet_Model.fit(batch_x_eval, batch_y_eval)

    print("Iteration {} ==> training ncc : {}, evaluate ncc : {} ".format(i+1, round(train_loss, 8), round(eval_loss, 8)))
    
    with open(txt_dir+'/data.txt','a') as f:    
      f.write(str(i)+" "+str(train_loss)+" "+str(eval_loss)+"\n")      

    if (i+1) % 1000 == 0:
      DIRNet_Model.deploy(config.tmp_dir, batch_x, batch_y)            
      DIRNet_Model.save(config.ckpt_dir)
      print("Model saved ... ...")

      for lable_id in range(2):
        result_i_dir = config.result_dir+"/{}".format(lable_id)
        DIRNet_Model.deploy(result_i_dir, batch_x_eval, batch_y_eval)   
        
     


if __name__ == "__main__":
  main()
