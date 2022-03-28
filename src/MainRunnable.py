import fastai; fastai.__version__
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
import warnings
from fastai.callbacks.hooks import *
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow
from PIL import Image,ImageEnhance
import warnings
import scipy.ndimage

warnings.filterwarnings('ignore')

#Preprocessed 5class Model load
learn=load_learner("./MainModel")
model=learn.model.eval()

#bounding box code -----------------------

class BBoxerwGradCAM():
    
    def __init__(self,learner,heatmap,image_path,resize_scale_list,bbox_scale_list):
        self.learner = learner
        self.heatmap = heatmap
        self.image_path = image_path
        self.resize_list = resize_scale_list
        self.scale_list = bbox_scale_list
        #smoothing the heatmap
        self.og_img, self.smooth_heatmap = self.heatmap_smoothing()
        #take contour boxes and formulate
        self.bbox_coords, self.poly_coords, self.grey_img, self.contours = self.form_bboxes()
        
    #Smoothening the heatmap
    def heatmap_smoothing(self):
        og_img = cv2.imread(self.image_path)
        heatmap = cv2.resize(self.heatmap, (self.resize_list[0],self.resize_list[1])) # Resizing
        og_img = cv2.resize(og_img, (self.resize_list[0],self.resize_list[1])) # Resizing
        heatmapshow = cv2.normalize(heatmap, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        
        return og_img, heatmapshow
    
    #Generate Bounding Boxes
    def form_bboxes(self):
        grey_img = cv2.cvtColor(self.smooth_heatmap, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grey_img,200,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        cv2.drawContours(self.og_img, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2_imshow("Bounding Box", self.og_img)
        cv2.imwrite("bBoxImage.png", self.og_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
	#contour creation
        for item in range(len(contours)):
            cnt = contours[item]
            if len(cnt)>20:
                #print(len(cnt))
                x,y,w,h = cv2.boundingRect(cnt)
                poly_coords = [cnt] #polygon coordinates
                
                x = int(x*self.scale_list[0]) 
                y = int(y*self.scale_list[1])
                w = int(w*self.scale_list[2])
                h = int(h*self.scale_list[3])

                return [x,y,w,h], poly_coords, grey_img, contours
            
            else: print("contour error (too small)")
                
    def get_bboxes(self):
        return self.bbox_coords, self.poly_coords

#gradcam code
class GradCam():
    @classmethod
    def from_interp(cls,learn,interp,img_idx,ds_type=DatasetType.Valid,include_label=False):
        # produce heatmap and xb_grad for pred label (and actual label if include_label is True)
        if ds_type == DatasetType.Valid:
            ds = interp.data.valid_ds
        elif ds_type == DatasetType.Test:
            ds = interp.data.test_ds
            include_label=False
        else:
            return None
        
        x_img = ds.x[img_idx]
        xb,_ = interp.data.one_item(x_img)
        #xb_img = Image(interp.data.denorm(xb)[0])
        probs = interp.preds[img_idx].numpy()

        pred_idx = interp.pred_class[img_idx].item() # get class idx of img prediction label
        hmap_pred,xb_grad_pred = get_grad_heatmap(learn,xb,pred_idx,size=xb_img.shape[-1])
        prob_pred = probs[pred_idx]
        
        actual_args=None
        if include_label:
            actual_idx = ds.y.items[img_idx] # get class idx of img actual label
            if actual_idx!=pred_idx:
                hmap_actual,xb_grad_actual = get_grad_heatmap(learn,xb,actual_idx,size=xb_img.shape[-1])
                prob_actual = probs[actual_idx]
                actual_args=[interp.data.classes[actual_idx],prob_actual,hmap_actual,xb_grad_actual]
        
        return cls(x_img,interp.data.classes[pred_idx],prob_pred,hmap_pred,xb_grad_pred,actual_args)
    
    @classmethod
    def from_one_img(cls,learn,x_img,label1=None,label2=None):
        pred_class,pred_idx,probs = learn.predict(x_img)
        label1= str(pred_class) if not label1 else label1
        
        xb,_ = learn.data.one_item(x_img)
        #xb_img = Image(learn.data.denorm(xb)[0])
        probs = probs.numpy()
        
        label1_idx = learn.data.classes.index(label1)
        hmap1,xb_grad1 = get_grad_heatmap(learn,xb,label1_idx,size=x_img.shape[-1])
        prob1 = probs[label1_idx]
        
        label2_args = None
        if label2:
            label2_idx = learn.data.classes.index(label2)
            hmap2,xb_grad2 = get_grad_heatmap(learn,xb,label2_idx,size=xb_img.shape[-1])
            prob2 = probs[label2_idx]
            label2_args = [label2,prob2,hmap2,xb_grad2]
            
        return cls(x_img,label1,prob1,hmap1,xb_grad1,label2_args)
    
    def __init__(self,x_img,label1,prob1,hmap1,xb_grad1,label2_args=None):
        self.x_img=x_img
        self.label1,self.prob1,self.hmap1,self.xb_grad1 = label1,prob1,hmap1,xb_grad1
        if label2_args:
            self.label2,self.prob2,self.hmap2,self.xb_grad2 = label2_args

def minmax_norm(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))
def scaleup(x,size):
    scale_mult=size/x.shape[0]
    upsampled = scipy.ndimage.zoom(x, scale_mult)
    return upsampled

# hook for Gradcam
def hooked_backward(m,xb,target_layer,clas):
    with hook_output(target_layer) as hook_a: #hook at last layer of group 0's output (after bn, size 512x7x7 if resnet34)
        with hook_output(target_layer, grad=True) as hook_g: # gradient w.r.t to the target_layer
            preds = m(xb)
            preds[0,int(clas)].backward() # same as onehot backprop
    return hook_a,hook_g

def clamp_gradients_hook(module, grad_in, grad_out):
    for grad in grad_in:
        torch.clamp_(grad, min=0.0)
        
# hook for guided backprop
def hooked_ReLU(m,xb,clas):
    relu_modules = [module[1] for module in m.named_modules() if str(module[1]) == "ReLU(inplace)"]
    with callbacks.Hooks(relu_modules, clamp_gradients_hook, is_forward=False) as _:
        preds = m(xb)
        preds[0,int(clas)].backward()
        
def guided_backprop(learn,xb,y):
    xb = xb.cuda()
    m = learn.model.eval();
    xb.requires_grad_();
    if not xb.grad is None:
        xb.grad.zero_(); 
    hooked_ReLU(m,xb,y);
    return xb.grad[0].cpu().numpy()

def get_grad_heatmap(learn,xb,y,size):
    '''
    Main function to get hmap for heatmap and xb_grad for guided backprop
    '''
    xb = xb.cuda()
    m = learn.model.eval();
    #here
    pred=learn.predict(img)
    print("All Predictions:\n",pred[2])
    cls=int(pred[1])
    print("Result: ")
    if(cls==0):
      print("COVID-19")
    elif(cls==1):
      print("Lung Opacity")
    elif(cls==2):
      print("Normal")
    elif(cls==3):
      print("Pneumonia")
    else:
      print("Tuberculosis")
    print(max(pred[2]*100))
    b, _ =learn.data.one_item(img,denorm=False)
    with hook_output(model[0]) as hook_a:
      with hook_output(model[0],grad=True) as hook_g:
        preds=model(b)
        preds[0,cls].backward()
    acts=hook_a.stored[0].cpu()
    grad=hook_g.stored[0][0].cpu()

    grad_chan=grad.mean(1).mean(1)
    hmap=((acts*grad_chan[..., None, None])).sum(0)
    _, ax=plt.subplots()
    img.show(ax)
    #----- make a line to save -------
    ax.imshow(hmap, alpha=0.4, extent=(0,224,224,0), interpolation='bicubic', cmap='jet')
    fig = ax.get_figure()
    fig.savefig("heatmap.png")
    hmap = np.where(hmap >= 0, hmap, 0)
    #upto here    
    xb_grad = guided_backprop(learn,xb,y) # (3,224,224)        
    #minmax norm the grad
    xb_grad = minmax_norm(xb_grad)
    hmap_scaleup = minmax_norm(scaleup(hmap,size)) # (224,224) 
    # multiply xb_grad and hmap_scaleup and switch axis
    xb_grad = np.einsum('ijk, jk->jki',xb_grad, hmap_scaleup) #(224,224,3)
    
    return hmap,xb_grad

# File Path

filepath="./Normal_Data/COVID-2244.png"
cvimage = cv2.imread(filepath,0)
cv2_imshow("MainImage",cvimage)
cv2.imwrite("claheSharp.png", cvimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
choice = input("Preprocessed? Y or N")
if choice == "N":
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    cl1 = clahe.apply(cvimage)
    cv2.imwrite("claheFile.png",cl1)
    image = Image.open("claheFile.png")
    enhancer=ImageEnhance.Sharpness(image)
    image=enhancer.enhance(4.0)
    img = np.array(image)
    cv2.imshow("Preprocessed Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("claheSharp.png",img)
img=open_image("claheSharp.png")
img.resize(224)

gcam = GradCam.from_one_img(learn,img)
#gcam.plot(plot_gbp = False)
gcam_heatmap = gcam.hmap1
#gcam_heatmap=mult
image_resizing_scale = [224,224]
bbox_scaling = [1,1,1,1]
bbox = BBoxerwGradCAM(learn,
                      gcam_heatmap,
                      filepath,
                      image_resizing_scale,
                      bbox_scaling)
