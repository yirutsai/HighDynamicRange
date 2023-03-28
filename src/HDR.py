import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import exifread
import math

def img_thresh_test(img):
  img_rgbsum = np.sum(img, 2)
  img_thresh = np.median(img_rgbsum)
  test_result = (img_rgbsum >= img_thresh)
  return test_result

def img_shift(img, pos):
  mat = np.float32([[1, 0, pos[0]],[0, 1, pos[1]]])
  shift_img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
  return shift_img

def img_xor(img1, img2):
  return np.sum(np.logical_xor(img1, img2))

def alignment(std_img, imgs, align_time):
  shift_points = [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]
  std_img_thresh = [img_thresh_test(std_img)]
  std_img_reshape = [std_img]
  aligned_img = []
  shift_pos = [0,0]

  for i in range(align_time):
    height, width = std_img_reshape[-1].shape[:2]
    std_img_reshape.append(cv2.resize(std_img_reshape[-1], (int(0.5*width), int(0.5*height)), interpolation = cv2.INTER_AREA))
    std_img_thresh.append(img_thresh_test(std_img_reshape[-1]))

  for img in imgs:
    img_reshape = [img]
    for i in range(align_time):
      height, width = img_reshape[-1].shape[:2]
      img_reshape.append(cv2.resize(img_reshape[-1], (int(0.5*width), int(0.5*height)), interpolation = cv2.INTER_AREA))
      
    for i in range(align_time-1, -1, -1):
      minxor_point = shift_points[0]
      min_xor = math.inf
      for point in shift_points:
        shift_pos = [pos/pow(2,i) for pos in shift_pos]
        align_img = img_shift(img_reshape[i], (shift_pos+point))
        align_img_thresh = img_thresh_test(align_img)
        xor = img_xor(std_img_thresh[i], align_img_thresh)
        if(xor<min_xor):
          minxor_point = point 
          min_xor = xor
      shift_pos = shift_pos + minxor_point*pow(2,i)

    aligned_img.append(img_shift(img, shift_pos))

  return aligned_img

def change(string):                 #OK
    if(string.find("/")!=-1):
        tmp = string.split("/")
        return float(tmp[0])/float(tmp[1])
    return float(string)

def read_imgs(dirpath):             #OK
    imgs = []
    expo_times = []
    for path in sorted(os.listdir(dirpath)):
        if(path[-3:]=="NEF"):
            print(path)
            f = open(dirpath+"./"+path,"rb")
            tag = (exifread.process_file(f))
            # print(tag.keys())
            exposure_time = tag['EXIF ExposureTime']
            # print(exposure_time)
            # print(type(exposure_time))
            expo_times.append(change(exposure_time.printable))
            f.close()
        elif(path[-3:]=="JPG"):
            print("path:",path)
            imgs.append(cv2.imread(dirpath+"./"+path,cv2.IMREAD_COLOR))
            img = cv2.imread(dirpath+"./"+path,cv2.IMREAD_COLOR)
            # f = open(dirpath+"./"+path,"rb")
            # tag = (exifread.process_file(f))
            # # print(tag.keys())
            # exposure_time = tag['EXIF ExposureTime']
            # # print(exposure_time)
            # # print(type(exposure_time))
            # expo_times.append(change(exposure_time.printable))
            # f.close()                     

    return imgs,expo_times

def get_weight(type = "linear",z_max = 255, z_min = 0):         
    hat = []
    z_mid = (z_min+z_max)//2
    for z in range(z_max+1):
        hat.append(z-z_min+1 if z<=z_mid else z_max-z+1)
    hat = np.array(hat)
    return hat

def sample(h,w,sample_rate):        #OK
    pos = []
    for i in range(1,sample_rate+1):
        for j in range(1,sample_rate+1):
            pos.append((i*int(h/sample_rate),j*int(w/sample_rate)))
    return pos

def solve(Z,B,lamda,W):
    pos_num,img_num , channel_num = Z.shape
    print("len(W):",len(W))
    print("len(B):",len(B))
    print("img_num:",img_num)
    print("pos_num:",pos_num)
    assert(len(W)==256 and len(B)==img_num)
    


    
    ans = []
    for channel in range(3):
        k=0
        A = np.zeros((img_num*pos_num+255,256+pos_num))
        b = np.zeros((A.shape[0],1))
        # print(A.shape)
        # print(b.shape)
        for i in range(pos_num):
            for j in range(img_num):
                
                # print(Z[i][j][channel])
                assert(int(Z[i][j][channel])==float(Z[i][j][channel]))
                
                wij = W[int(Z[i][j][channel])]
                A[k][int(Z[i][j][channel])] = wij
                A[k][256+i] = -wij

                b[k][0] = wij*B[j]
                k += 1

        A[k,128] = 1
        k+=1

        for i in range(0,254):
            A[k][i] = lamda *W[i+1]
            A[k][i+1] = -2*lamda*W[i+1]
            A[k][i+2] = lamda*W[i+1]
            k+=1
    
        X = np.linalg.lstsq(A, b, rcond = None)[0].ravel()
        ans.append((X[:256]))
    return ans



def I2Z(images,pos):
    z = np.zeros((len(pos),len(images),3))
    for channel in range(3):
        for i,img in enumerate(images):
            for j,(x,y) in enumerate(pos):
                z[j][i][channel] = img[x][y][channel]
    return z

def rad(imgs,b,W,g):
    rads = []
    imgs = np.array(imgs)
    print("images.shape:",imgs.shape)
    for channel in range(3):
        E =[]
        for idx,img in enumerate(imgs):
            E.append(g[channel][img[:,:,channel]]-b[idx])
        # print("imgs.shape:",W[imgs[:,:,:,channel]])

        rads.append(np.average(E,axis = 0,weights = W[imgs[:,:,:,channel]]))

    return rads
def produce_rad_map(rads,output_dirpath,colors):
    for idx,rad in enumerate(rads):
        plt.figure()
        plt.imshow(rad, cmap = plt.cm.plasma)
        plt.colorbar()
        plt.savefig(output_dirpath+"/radiance_map_"+str(idx)+".png")
        plt.close()

def HDR(args):
    colors = ['r','g','b']
    input_path = os.path.dirname(__file__)+"/../"+args.input_dirpath
    # print(input_path)
    imgs,expo_times = read_imgs(input_path)
    print("read",len(imgs),"images")
    print("every image's shape",imgs[0].shape)
    print("exposure times are",expo_times)
    W = get_weight(type = args.weight)
    if(args.align):
        imgs = alignment(imgs[0],imgs,args.alignTime)
    print(len(imgs))
    
    B = np.log(expo_times,dtype = np.float32)
    # print(B.shape)
    
    img_h, img_w, _ = imgs[0].shape
    sample_rate = args.sampling
    sample_pos = sample(img_h,img_w,sample_rate)

    Z = I2Z(imgs, sample_pos)
    print("Z.shape=",Z.shape)
    plt.figure()
    g = solve(Z, B, args.lamda, W)


    print("len(g)=",len(g))

    rads = rad(imgs,B,W,g)
    for channel in range(3):
        plt.plot(g[channel],range(256),colors[channel])
    # plt.show()
    

    output_path = os.path.dirname(__file__)+'/../'+args.output_dirpath+"/"+args.input_dirpath.split("/")[-1]+"_lambda-"+str(args.lamda)+"_"+"weight-"+args.weight+"_sampling-"+str(args.sampling)
    if(args.align == True):
        output_path +="_alignTime-"+str(args.alignTime)
    os.makedirs(output_path, exist_ok=True)
    produce_rad_map(rads,output_path,colors)

    radiance_map_final = np.transpose(np.exp(np.stack(rads)),(1,2,0))
    print(radiance_map_final.shape)
    tonemap = cv2.createTonemap(2.2)
    ldr = tonemap.process(radiance_map_final.astype(np.float32))
    # ldr = np.power(ldr,1.5)*80
    # print(np.max(ldr))
    # print(np.min(ldr))
    cv2.imwrite(output_path+'/ldr.png', ldr * 255)
    cv2.imwrite(output_path+"/produce.hdr",radiance_map_final.astype(np.float32))
    plt.savefig(output_path+"/expo.png")

    