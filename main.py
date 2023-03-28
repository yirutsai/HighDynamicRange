import os
from argparse import ArgumentParser
import src.HDR

if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dirpath",type =str)
    parser.add_argument("output_dirpath",type = str)
    parser.add_argument("--weight",type=str,default = "linear")
    parser.add_argument("--lamda", type = int, default = 15)
    parser.add_argument("--align",action = "store_true")
    parser.add_argument("--sampling",type = int,default= 21)
    
    parser.add_argument("--alignTime",type = int,default= 5)
    
    args = parser.parse_args()
    src.HDR.HDR(args)