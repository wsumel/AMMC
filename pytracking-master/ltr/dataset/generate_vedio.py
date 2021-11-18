import numpy  as np 
import random 
import cv2 as cv
from random import choice
import math 
import matplotlib.pyplot as plt

def draw_gt(im,gt=None,savefig_path='./',clip_box=None):
    
    dpi = 80.0
    figsize = (im.shape[1]/dpi, im.shape[0]/dpi)

    fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(im)

    if clip_box is not None:
        clip_rect = plt.Rectangle([clip_box[0],clip_box[1]],clip_box[2],clip_box[3],
                linewidth=3, edgecolor="r", zorder=1, fill=False)
        ax.add_patch(clip_rect)
    
    gt_rect = plt.Rectangle([gt[0],gt[1]],gt[2],gt[3],
        linewidth=3, edgecolor="g", zorder=1, fill=False)
    ax.add_patch(gt_rect)

    # plt.show()
    fig.savefig(savefig_path,dpi=dpi)
    plt.close()

class Motion:

    def __init__(self,im,init_gt):
        self.init_gt = init_gt
        init_gt = self.clip_init_gt(init_gt,[im.shape[1],im.shape[0]])
        self.im = im
        self.gt = init_gt
        self.scene_sz = [im.shape[1],im.shape[0]]

        self.init_center = init_gt[:2] + init_gt[2:]/2
        self.init_sz = init_gt[2:]
        self.aspect_f = init_gt[2]/init_gt[3]
        self.scale_f = 1.2

        self.target = None
        self.scene = None

        self.size_range =  self.set_size_range()
        self.aspect_range = (init_gt[2]/init_gt[3])* np.array([0.5,2]) 
        self.brightness_change_direction = 1

        self.cm_sg = 0.2
        self.oc_sg = 0.3
        self.projective_sg = 0.2

        self.redirection_p = 0.4
        self.inside_p = 0.7
        # self.low_resolution_p = 0.1
        self.aspect_p = 0.3
        self.scale_p = 0.5
        self.out_of_view = 0
        self.move_p=0.8
        self.brightness_f = 1
        self.rotate_p = 0.2
        self.motion_blur_p = 0.1
        self.blur_p = 0.3
        self.projective_p = 0.3
        self.part_occluded_p = 0.65
        self.camera_motion_p = 0.3
        self.flip_p = 0.5
        self.motion_p = 0.8
        self.target,self.scene = self.split_target_scene(im,self.gt)

    def clip_init_gt(self,init_gt,im_shape):
        init_gt[0] = int(init_gt[0])
        init_gt[1] = int(init_gt[1])
        init_gt[2] = math.ceil(init_gt[2])
        init_gt[3] = math.ceil(init_gt[3])
        x1,y1 = init_gt[:2]
        x2,y2 = init_gt[:2] + init_gt[2:]

        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,im_shape[0])
        y2 = min(y2,im_shape[1])

        return np.array([x1,y1,x2-x1,y2-y1]).astype(np.int)

    def set_size_range(self):
        if np.prod(self.gt[2:]) < 32 * 32:
            return np.prod(self.gt[2:]) * np.array([1,2])
        elif np.prod(self.gt[2:]) < np.prod(self.scene_sz) * 0.15:
            return np.prod(self.gt[2:]) * np.array([0.5,2])
        else:
            return np.array([np.prod(self.scene_sz)* 0.15 ,np.prod(self.scene_sz)* 0.4])

    def flip(self,im,gt):
        if self.roll(self.flip_p) == False:
            return im,gt
        gt[0] = self.scene_sz[0] - gt[0] - gt[2]
        im = np.fliplr(im)
        return im,gt 


    def split_target_scene(self,im,gt):
        
        im_new = im.copy()
        im_new.flags.writeable = True

        x1,y1 = gt[:2]
        x2,y2 = gt[:2] + gt[2:]

        target = np.copy(im_new[(y1):(y2),(x1):(x2),:])

        mask = np.zeros(im.shape[:2],dtype=np.uint8) 
        mask[(y1):(y2),(x1):(x2)] = 255
        im_new = cv.inpaint(im_new,mask,3,cv.INPAINT_NS)
        
        scene = np.copy(im_new)
        return target,scene
    
    def roll(self,p):
        return random.uniform(0,1) <p   

    def brightness_change(self,im):
        alpha=1 
        if self.brightness_f > 112 or self.brightness_f< -122: 
            self.brightness_change_direction = -self.brightness_change_direction
        self.brightness_f = self.brightness_f + 2*self.brightness_change_direction
        h,w,c = im.shape
        blank = np.zeros([h,w,c],im.dtype)
        new_im = cv.addWeighted(im,alpha,blank,1-c,self.brightness_f)
        return new_im

    def camera_motion(self,im,gt):
        if self.roll(self.camera_motion_p) ==False:
            return im ,gt
        shape = im.shape
        while(True):
            mw = int((np.random.random() * 2 - 1.0)  * shape[1]* self.cm_sg)
            mh = int((np.random.random() * 2 - 1.0)  * shape[0]* self.cm_sg)
            mat_translation=np.float32([[1,0,mw],[0,1,mh]])
            cx,cy = gt[:2] + gt[2:]/2
            if 0 < (cx+mw) < shape[1] and 0 < (cy+mh) < shape[0]:
                gt[0] = gt[0]+mw
                gt[1] = gt[1]+mh
                break 

        return cv.warpAffine(im,mat_translation,(shape[1],shape[0])),gt

    def partially_occluded(self,mask):
        oc_sg_h = np.random.random() + 0.001
        oc_sg_w = (self.oc_sg / oc_sg_h) 
        oc_sg_w = min(1,oc_sg_w) 
        oc_sg_w = oc_sg_w * np.random.random()
        och = int(np.random.random()*oc_sg_h*mask.shape[0])
        ocw = int(np.random.random()*oc_sg_w*mask.shape[1])
        lh = int((mask.shape[0]-och)*np.random.random())
        lw = int((mask.shape[1]-ocw)*np.random.random())
        #print(lh,lw,och,ocw)
        mask[lh:lh+och,lw:lw+ocw,:] = 0
        return mask   
   
    def occlustion_target(self,mask):
        if self.roll(self.part_occluded_p) == False:
            return mask
        oc_mask = self.partially_occluded(mask)
        return oc_mask

    def rotate_target(self,im):
        if self.roll(self.rotate_p) == False:
            return self.target,self.gt[2:],True

        #random angle
        angle = np.random.uniform(5,60)
        if self.roll(0.5) == True:
            angle = -angle

        # new size that contain all target
        w,h = self.scene_sz
        H = cv.getRotationMatrix2D((w/2,h/2),angle,1)
        sin_alpha = math.fabs(math.sin(math.pi * angle/180))
        cos_alpha = math.fabs(math.cos(math.pi * angle/180))
        height_new = int(w*sin_alpha+h*cos_alpha)
        width_new = int(h*sin_alpha+w*cos_alpha)
        H[0,2] +=(width_new-w)/2
        H[1,2] +=(height_new-h)/2
        #rotate
        im = cv.warpAffine(im, H, (width_new,height_new))   
        #generate mask
        x1,y1 = self.gt[:2]
        x2,y2 = self.gt[:2] + self.gt[2:]
        x_list = [x1,x2,x2,x1]
        y_list = [y1,y1,y2,y2]
        verts=[]
        for x0,y0 in zip(x_list,y_list):
            Q= np.dot(H,np.array([[[x0],[y0],[1]]]))
            verts.append([ int(Q[0][0][0]),int(Q[1][0][0]) ])
        verts = np.array(verts)
        left = np.min(verts,axis=0)
        right = np.max(verts,axis=0)
        # target_gt = np.concatenate((left,right-left),axis=0)
        # draw_gt(im,target_gt,'./test.jpg')
        target = np.copy(im[(left[1]):(right[1]),(left[0]):(right[0]),:])
        # mdst = np.sum(target,axis=2)
        # mh,mw = mdst.shape[:2]
        # tmask = np.where(mdst>0,1,0)
        # mask = tmask.reshape(mh,mw,1)
        return  target,right-left,False

    def projective(self,im):
        if self.roll(self.projective_p) == False or self.gt[2]<4 or self.gt[3]<4:
            return self.target,self.gt[2:]
            
        cols,rows = self.gt[2:]
        p1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
        p1 = (p1 + self.gt[:2]).astype(np.float32)
        hyp = self.projective_sg
        ax = np.random.random()*2*hyp
        ay = np.random.random()*2*hyp
        bx = np.random.random()*2*hyp + 1
        by = np.random.random()*2*hyp
        cx = np.random.random()*2*hyp
        cy = np.random.random()*2*hyp + 1
        dx = np.random.random()*2*hyp + 1
        dy = np.random.random()*2*hyp + 1
        pro_gt = [[ax*(cols-1),ay*(rows-1)], [bx*(cols-1),by*(rows-1)], [cx*(cols-1),cy*(rows-1)], [dx*(cols-1),dy*(rows-1)]] 
        p2 = np.float32(pro_gt)
        p2 = (p2 + self.gt[:2]).astype(np.float32)
        M = cv.getPerspectiveTransform(p1,p2)
        dst = cv.warpPerspective(im, M, (int(self.scene_sz[0]*1.2), int(self.scene_sz[1]*1.2)))
        #draw_gt(dst,np.array([0,0,0,0]),'./dst.jpg')
        dst = dst[int(min(p2[0][1],p2[1][1])):int(max(p2[2][1],p2[3][1])),int(min(p2[0][0],p2[2][0])):int(max(p2[1][0],p2[3][0])),:]
        
        return dst,np.array([dst.shape[1],dst.shape[0]])

    def generate_mask(self,im_patch):
        mdst = np.sum(im_patch,axis=2)
        mh,mw = mdst.shape[:2]
        tmask = np.where(mdst>0,1,0)
        return tmask.reshape(mh,mw,1)

    def motion_blur(self,im_patch,degree=10,angle=20):
        if self.roll(self.motion_blur_p) == True:
            M = cv.getRotationMatrix2D((degree/2, degree/2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (degree, degree))
        
            motion_blur_kernel = motion_blur_kernel / degree        
            im_patch = cv.filter2D(im_patch, -1, motion_blur_kernel)
            # convert to uint8
            cv.normalize(im_patch, im_patch, 0, 255, cv.NORM_MINMAX)
            im_patch = np.array(im_patch, dtype=np.uint8)
        return im_patch
    
    def blur(self,im_patch):
        if self.roll(self.blur_p) == True:
            im_patch = cv.GaussianBlur(im_patch,(5,5),0)
        return im_patch
    
    def generate_center_point(self):
        if self.roll(self.motion_p) == False:
            return self.init_center
        return np.random.uniform(self.init_sz/2,self.scene_sz-self.init_sz/2)

    def scale_change(self,obj_size):
        if self.roll(self.scale_p) == False and np.prod(obj_size) < self.size_range[1]:
            return obj_size
        resize_range = np.sqrt(self.size_range / np.prod(obj_size))
        scale = random.uniform(resize_range[0],resize_range[1])
        return obj_size * scale 
    
    def aspect_change(self,obj_size):
        if self.roll(self.aspect_p) == False:
            return obj_size
        ratio_range = np.sqrt(self.aspect_range / (obj_size[0]/obj_size[1]))
        ratio = random.uniform(ratio_range[0],ratio_range[1])
        aspect_f = [ratio,1/ratio]

        obj_size = obj_size * aspect_f
        if obj_size[0] / obj_size[1] < self.aspect_range[0] or obj_size[0] / obj_size[1] >self.aspect_range[1]:
            print('here')
        return obj_size
    
    def change_target_size(self,im_patch,new_size):
        try:
            return cv.resize(im_patch,(new_size[0],new_size[1]))
        except:
            print(self.gt,new_size,self.init_gt)

    def compose_target_scene(self,changed_patch,changed_scene,new_gt,mask):
        
        x1,y1 = [0,0]
        x2,y2 = new_gt[2:]
        
        scene_gt = np.array([0,0,self.scene_sz[0],self.scene_sz[1]])
        gt_x1 = np.maximum(scene_gt[0], new_gt[0])
        gt_x2 = np.minimum(scene_gt[0]+scene_gt[2], new_gt[0]+new_gt[2])
        gt_y1 = np.maximum(scene_gt[1], new_gt[1])
        gt_y2 = np.minimum(scene_gt[1]+scene_gt[3], new_gt[1]+new_gt[3])

        x1 = gt_x1 - new_gt[0]
        x2 = gt_x2 - new_gt[0]
        y1 = gt_y1 - new_gt[1]
        y2 = gt_y2 - new_gt[1]

        im_new = np.copy(changed_scene)
        patch_new = np.copy(changed_patch)
        mask = mask[y1:y2,x1:x2,]
        try:
            im_new[gt_y1:gt_y2,gt_x1:gt_x2:,]  = np.multiply(patch_new[y1:y2,x1:x2,],mask) + np.multiply(im_new[gt_y1:gt_y2,gt_x1:gt_x2:,],1-mask)
        except:
            print('error exist')  
        return im_new
    
    def checkgt(self,gt):
        cx,cy = gt[:2]+gt[2:]
        if gt[0] < 0:
            gt[2] = gt[2] + gt[0]
            gt[0] = 0
        if gt[1] < 0:
            gt[3] = gt[3] + gt[1]
            gt[1] = 0
        gt[2] = min(gt[2],self.scene_sz[0]-gt[0])
        gt[3] = min(gt[3],self.scene_sz[1]-gt[1]) 
        return gt

    def __call__(self):
        #copy value
        new_size = np.copy(self.gt[2:])
        new_center = np.copy(self.gt[:2] + self.gt[2:]/2)
        tmp_target = np.copy(self.target)
        tmp_scene = np.copy(self.scene)
        if self.gt[2]> 4 and self.gt[3] > 4:
            # rotate
            tmp_target,new_size,rp_flag = self.rotate_target(np.copy(self.im))
            # draw_gt(tmp_target,np.array([0,0,0,0]),savefig_path='./a.jpg')
            # projective
            if rp_flag:
                tmp_target,new_size = self.projective(np.copy(self.im)) 

            # aspect and scale change
            new_size = self.aspect_change(new_size)
      
        new_size = self.scale_change(new_size)
        new_size = new_size.astype('int')
        tmp_target = self.change_target_size(tmp_target,new_size)       
        mask = self.generate_mask(tmp_target)
        # motion (leave)
        new_center = self.generate_center_point()
        #blur target
        tmp_target = self.motion_blur(tmp_target)
        # occlustion
        mask = self.occlustion_target(mask)

        # low resolution random
        # out of view(part or full) random
        # background clutters (similar color or texture)

        # update state
        new_gt = np.concatenate((new_center - new_size/2,new_size)).astype('int') 
        if np.prod(new_size)/np.prod(self.scene_sz) > 0.4:
            print('error',np.prod(new_size)/np.prod(self.scene_sz))
        # print(np.prod(new_size)/np.prod(self.scene_sz))
    
        #compose new img
        new_im = self.compose_target_scene(tmp_target,tmp_scene,new_gt,mask)
        
        # illumination change
        new_im = self.brightness_change(new_im)
        # blur im
        new_im = self.blur(new_im)    
        # Abrupt motion of camera
        
        new_gt = self.checkgt(new_gt)

        new_im,new_gt = self.camera_motion(new_im,new_gt)
        new_gt = self.checkgt(new_gt)
        # show_image(new_im)
        
        #flip image
        new_im,new_gt = self.flip(new_im,new_gt)

        return new_im,new_gt

