# File: custom_modules/pneumonia_classifier.py
import torch
import torch.nn as nn
# MMDetection/MMCV ফ্রেমওয়ার্কের প্রয়োজনীয় ইমপোর্ট
from mmcv.runner import BaseModule
from mmdet.models.builder import CLASSIFIERS, build_backbone 
from torch.nn import CrossEntropyLoss

# আমাদের কাস্টম ক্লাসিফায়ারকে মডিউল রেজিস্ট্রি-তে যুক্ত করা হলো
@CLASSIFIERS.register_module()
class PneumoniaClassifier(BaseModule):
    """
    Complete model for Pneumonia X-ray Binary Classification 
    using ResNet_FDConv backbone.
    """
    def __init__(self, backbone_cfg, num_classes=2, init_cfg=None):
        super().__init__(init_cfg)
        
        # ১. ব্যাকবোন লোড করা: ResNet_FDConv ব্যবহার করা হবে
        self.backbone = build_backbone(backbone_cfg)
        
        # ResNet-এর শেষ স্টেজের ফিচার ডাইমেনশন (যেমন ResNet-50 এর জন্য 2048)
        # ধরে নিলাম আপনার ResNet_FDConv ক্লাসে feat_dim প্যারামিটার আছে। 
        # যদি না থাকে, ResNet-50 এর জন্য 2048 এবং ResNet-34 এর জন্য 512 ব্যবহার করুন।
        feat_dim = self.backbone.feat_dim 
        self.num_classes = num_classes
        
        # ২. ক্লাসিফিকেশন হেড তৈরি করা
        self.fc_head = nn.Sequential(
            # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),                 # (B, C, 1, 1) -> (B, C)
            # Linear Layer: (B, feat_dim) -> (B, num_classes=2)
            nn.Linear(feat_dim, num_classes) 
        )
        
        # ৩. লস ফাংশন
        self.loss_cls = CrossEntropyLoss()

    def forward_train(self, img, gt_label):
        """ট্রেনিং এর জন্য ফরওয়ার্ড পাস (লস গণনা করে)"""
        # ব্যাকবোন থেকে ফিচার এক্সট্র্যাক্ট করা
        features = self.backbone(img)
        # ResNet সাধারণত Tuple হিসেবে ফিচার দেয়, শেষ স্টেজটি নেওয়া হলো
        x = features[-1] 
        
        # হেড থেকে প্রেডিকশন
        cls_score = self.fc_head(x)
        
        # লস গণনা
        loss = self.loss_cls(cls_score, gt_label.squeeze()) 
        
        return dict(loss_cls=loss)

    def forward_test(self, img):
        """ইনফারেন্স এর জন্য ফরওয়ার্ড পাস (স্কোর রিটার্ন করে)"""
        features = self.backbone(img)
        x = features[-1]
        cls_score = self.fc_head(x)
        return cls_score

    def forward(self, img, gt_label=None, return_loss=True, **kwargs):
        """ট্রেনিং বা টেস্টিং এর উপর ভিত্তি করে সঠিক ফাংশন কল করা"""
        if return_loss and gt_label is not None:
            return self.forward_train(img, gt_label)
        else:
            return self.forward_test(img)
