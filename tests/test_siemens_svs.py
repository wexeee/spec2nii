import pytest
import numpy as np
import os.path as op
import subprocess
from fsl_mrs.utils import mrs_io
from PIL import Image,ImageDraw,ImageFont  

# Data paths
siemens_path = op.join(op.dirname(__file__),'spec2nii_test_data','Siemens')
vb_path = op.join(siemens_path,'VBData')
ve_path = op.join(siemens_path,'VEData')

# SVS VB data
svs_data_names_vb = ['C_T15_S10_10',
                 'S_C10_S5_10',
                 'T_C15_S10_10',
                 'iso_tra']

svs_data_twix_vb =['Twix/meas_MID151_svs_se_C_T15_S10_10_FID108741.dat',
                'Twix/meas_MID149_svs_se_S_C10_S5_10_FID108739.dat',
                'Twix/meas_MID147_svs_se_T_C15_0_S10_0_10_FID108737.dat',
                'Twix/meas_MID153_svs_se_Tra_sat_FID108743.dat']

svs_Data_dicom_vb = ['DICOM/svs_se_C>T15>S10_10_12_1',
                  'DICOM/svs_se_S>C10>S5_10_11_1',                    
                  'DICOM/svs_se_T>C15.0>S10.0_10_10_1',
                  'DICOM/svs_se_Tra_sat_13_1']

screen_shots_vb = ['Screenshots/svs_se_C_T15_S10_10.png',
                'Screenshots/svs_se_S_C10_S5_10.png',
                'Screenshots/svs_se_T_C15_0_S10_0_10.png',                
                'Screenshots/svs_se_Tra_sat.png']

# SVS VE data
svs_data_names_ve = ['C_T15_S10_10',
                 'S_C10_S5_10',
                 'T_C15_S10_10',
                 'iso_tra']

svs_data_twix_ve =['Twix/meas_MID00240_FID62745_svs_se_c_t15_s10_R10.dat',
                'Twix/meas_MID00238_FID62743_svs_se_s_c10_t5_R10.dat',
                'Twix/meas_MID00235_FID62740_svs_se_t_c15_s10_R10.dat',
                'Twix/meas_MID00242_FID62747_svs_se_iso_tra_sat.dat']

svs_Data_dicom_ve = ['DICOM/svs_se_c>t15>s10_R10_12_1',
                  'DICOM/svs_se_s>c10>t5_R10_11_1',                    
                  'DICOM/svs_se_t>c15>s10_R10_10_1',
                  'DICOM/svs_se_iso_tra_sat_13_1']

screen_shots_ve = ['Screenshots/svs_se_c_t15_s10_R10.png',
                'Screenshots/svs_se_s_c10_t5_R10.png',
                'Screenshots/svs_se_t_c15_s10_R10.png',                
                'Screenshots/svs_se_iso_tra_sat.png']


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (max(im1.width,im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

# Test siemens VB svs
def test_VB(tmp_path):
    sub_images = []
    for idx,(f_t,f_d,name) in enumerate(zip(svs_data_twix_vb,svs_Data_dicom_vb,svs_data_names_vb)):
        # Convert twix
        subprocess.check_call(['spec2nii','twix','-e','image',
                                '-f',name+'_t',
                                '-o',tmp_path,
                                '-j',op.join(vb_path,f_t)])
        # Convert DICOM
        subprocess.check_call(['spec2nii','dicom',
                                '-f',name+'_d',
                                '-o',tmp_path,
                                '-j',op.join(vb_path,f_d)])

        # Make fsleyes rendering
        subprocess.check_call(['pythonw','/Users/wclarke/opt/miniconda3/envs/fsl_mrs/bin/fsleyes',
                            'render','-of',op.join(tmp_path,f'svs_{idx}.png'),
                            '-vl','95','101','97',
                            '-hc',op.join(vb_path,'T1.nii.gz'),
                            op.join(tmp_path,name+'_t_Set000.nii.gz'),'-a','50','-cm','red',
                            op.join(tmp_path,name+'_d_000.nii.gz'),'-a','50','-cm','blue'])

        _,header_t = mrs_io.read_FID(op.join(tmp_path,name+'_t_Set000.nii.gz'))
        _,header_d = mrs_io.read_FID(op.join(tmp_path,name+'_d_000.nii.gz'))

        assert np.allclose(header_t['nifti'].affine,header_d['nifti'].affine)

        fsl_ss = Image.open(op.join(tmp_path,f'svs_{idx}.png'))
        width, height = fsl_ss.size
        fsl_ss_cropped = fsl_ss.crop((0,180,width,height-180))
        
        ss = Image.open(op.join(vb_path,screen_shots_vb[idx]))
        width, height = ss.size
        ss_cropped = ss.crop((0,50,width-50,height/2-80))
        
        fsl_ss_cropped=fsl_ss_cropped.resize(ss_cropped.size)
        sub_images.append(get_concat_v(fsl_ss_cropped, ss_cropped))        

    final_img = Image.new('RGB', (sub_images[0].width*2, sub_images[0].height*2))
    draw = ImageDraw.Draw(final_img)
    for idx,si in enumerate(sub_images):
        c = idx%2
        r = int(idx/2)
        final_img.paste(si, (si.width*r, si.height*c))
        draw.text((10+si.width*r, 10+si.height*c),svs_data_names_vb[idx],(255,0,0))

    final_img.save(op.join(op.dirname(__file__),'svs_vb.png'))

def test_VE(tmp_path):
    sub_images = []
    for idx,(f_t,f_d,name) in enumerate(zip(svs_data_twix_ve,svs_Data_dicom_ve,svs_data_names_ve)):
        # Convert twix
        subprocess.check_call(['spec2nii','twix','-e','image',
                                '-f',name+'_t',
                                '-o',tmp_path,
                                '-j',op.join(ve_path,f_t)])
        # Convert DICOM
        subprocess.check_call(['spec2nii','dicom',
                                '-f',name+'_d',
                                '-o',tmp_path,
                                '-j',op.join(ve_path,f_d)])

        # Make fsleyes rendering
        if idx == 3:
            x,y,z = '57','48','37'
        else:
            x,y,z = '53','67','56'
        subprocess.check_call(['pythonw','/Users/wclarke/opt/miniconda3/envs/fsl_mrs/bin/fsleyes',
                            'render','-of',op.join(tmp_path,f'svs_{idx}.png'),
                            '-vl',x,y,z,
                            '-hc',op.join(ve_path,'T1.nii.gz'),
                            op.join(tmp_path,name+'_t_Ave000.nii.gz'),'-a','50','-cm','red',
                            op.join(tmp_path,name+'_d_000.nii.gz'),'-a','50','-cm','blue'])

        _,header_t = mrs_io.read_FID(op.join(tmp_path,name+'_t_Ave000.nii.gz'))
        _,header_d = mrs_io.read_FID(op.join(tmp_path,name+'_d_000.nii.gz'))

        assert np.allclose(header_t['nifti'].affine,header_d['nifti'].affine)

        fsl_ss = Image.open(op.join(tmp_path,f'svs_{idx}.png'))
        width, height = fsl_ss.size
        fsl_ss_cropped = fsl_ss.crop((0,180,width,height-180))
        
        ss = Image.open(op.join(ve_path,screen_shots_ve[idx]))
        width, height = ss.size
        ss_cropped = ss.crop((0,50,width-50,height/2-80))
        
        fsl_ss_cropped=fsl_ss_cropped.resize(ss_cropped.size)
        sub_images.append(get_concat_v(fsl_ss_cropped, ss_cropped))        

    final_img = Image.new('RGB', (sub_images[0].width*2, sub_images[0].height*2))
    draw = ImageDraw.Draw(final_img)
    for idx,si in enumerate(sub_images):
        c = idx%2
        r = int(idx/2)
        final_img.paste(si, (si.width*r, si.height*c))
        draw.text((10+si.width*r, 10+si.height*c),svs_data_names_ve[idx],(255,0,0))

    final_img.save(op.join(op.dirname(__file__),'svs_ve.png'))

def test_data_size(tmp_path):
    for idx,(f_t,f_d,name) in enumerate(zip(svs_data_twix_vb,svs_Data_dicom_vb,svs_data_names_vb)):
        # Convert twix
        subprocess.check_call(['spec2nii','twix','-e','image',
                                '-f',name+'_t',
                                '-o',tmp_path,
                                '-j',op.join(vb_path,f_t)])

        data_t,header_t = mrs_io.read_FID(op.join(tmp_path,name+'_t_Set000.nii.gz'))

        assert data_t.shape == (1056,32)

    for idx,(f_t,f_d,name) in enumerate(zip(svs_data_twix_ve,svs_Data_dicom_ve,svs_data_names_ve)):
        # Convert twix
        subprocess.check_call(['spec2nii','twix','-e','image',
                                '-f',name+'_t',
                                '-o',tmp_path,
                                '-j',op.join(ve_path,f_t)])
        
        data_t,header_t = mrs_io.read_FID(op.join(tmp_path,name+'_t_Ave000.nii.gz'))

        assert data_t.shape == (1056,32)
