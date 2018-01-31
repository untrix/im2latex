#!/usr/bin/env python2
"""
This file executes disp_alpha.ipynb in bulk, once for each of the images below. It then exports the notebook into
HTML in the current folder. Run it in a folder such as gallery/I2L-NOPOOL along with the companion disp_alpha.ipynb.
"""
import sys, os
from multiprocessing import Pool

def do(image_name):
    def rmtail(s, t):
        return s.rsplit(t, 1)[0]

    # import nbformat
    # from nbconvert.preprocessors import ExecutePreprocessor

    os.environ['image_name'] = repr(image_name)
    os.putenv('image_name', image_name)

    # with open('disp_alpha.ipynb') as f:
    #     nb = nbformat.read(f, as_version=4)
    # ep = ExecutePreprocessor(timeout=600)  # kernel_name='python2')
    # ep.preprocess(nb)

    command = 'jupyter nbconvert --to HTML --output alpha_%s_gray --execute --ExecutePreprocessor.timeout=300 disp_alpha.ipynb'%rmtail(image_name, '.png')
    print('Executing command: %s'%command)
    os.system(command)

if __name__ == '__main__':
    if 'image_name' in os.environ:
        print('Processing image %s'%os.environ['image_name'])
        exit(0)

    p = Pool(2)
    I2L_NOPOOL_50_MATCHED = [
        u'6de537f98f51a70.png',
        u'e151d0cb6a1f4b8.png',
        u'125e0edbdc14c16.png',
        u'c1a595cf0e1b410.png',
        u'976c67c09595d48.png',
        u'48e151a0a2e1d66.png',
        u'eb4edff43972a77.png',
        u'f535e2d3ffd72a9.png',
        u'fbf3c74e173ede6.png',
        u'b727765af13988d.png',
        u'c236ef8f2d69db4.png',
        u'17806d8a43ed4d7.png',
        u'7bf25eec600c770.png',
        u'd67b0016af15368.png',
        u'beac5a98ad0bba3.png',
        u'6589b8b41dec5f5.png',
        u'c53968dbdf5e491.png',
        u'f7df71e09e679fa.png',
        u'88085cbe4db62f4.png',
        u'a4069d6109fdb32.png',
        u'7e7c82bcbbab14d.png',
        u'4cab7f4e7119975.png',
        u'ee3f8d415a17042.png',
        u'09c406611c97ca6.png',
        u'acc6b030ec1db54.png',
        u'5bde325cdc5c9fb.png',
        u'fc51f4f92be6b9e.png',
        u'831233abfc981bb.png',
        u'2af02fe9dda544b.png',
        u'dc311ef87140544.png',
        u'62d52e5875f15f2.png',
        u'082d6f67587ff53.png',
        u'be5020af1c11fb0.png',
        u'0ebe66af564fdea.png',
        u'8f17277609baf0d.png',
        u'cd7ee25bb44ee96.png',
        u'bde00b1efb71c8f.png',
        u'1637deef28fa753.png',
        u'ba84027cf12d913.png',
        u'ca7098dc8853675.png',
        u'5be77b312bfa0c1.png',
        u'9afabb69abb8665.png',
        u'e75a0c252c98431.png',
        u'05a32153f52b845.png',
        u'c450aeeee50eacb.png',
        u'8f249bcfcbd0d4a.png',
        u'c9908dd9001ae2a.png',
        u'8a7278fd1af0571.png',
        u'780ce6e35d2dfb2.png',
        u'04237c2640a6ef2.png']

    I2L_STRIPS_50_MATCHED = [
        u'4ef63353075e5b6.png',
     u'6de537f98f51a70.png',
     u'9dc9caeac24960d.png',
     u'125e0edbdc14c16.png',
     u'48e151a0a2e1d66.png',
     u'd4b25f217be4cca.png',
     u'7e7c82bcbbab14d.png',
     u'c236ef8f2d69db4.png',
     u'fbf3c74e173ede6.png',
     u'e8bd11a6b2feacf.png',
     u'b727765af13988d.png',
     u'7bf25eec600c770.png',
     u'21b2c45e268829b.png',
     u'f7df71e09e679fa.png',
     u'd67b0016af15368.png',
     u'6201fd941a8d4da.png',
     u'6f3d3d2ed89345d.png',
     u'beac5a98ad0bba3.png',
     u'136ca940c9932d4.png',
     u'cda328a07cba902.png',
     u'7fec9f1799b13ec.png',
     u'6147055797ca25d.png',
     u'938f5c3d05f5cf4.png',
     u'eebbeeddab4c0af.png',
     u'f16ea5d12d68b60.png',
     u'0734f11afe9aa90.png',
     u'186678817078727.png',
     u'2590ff270553f09.png',
     u'ee3f8d415a17042.png',
     u'4cab7f4e7119975.png',
     u'e99ef7e83d7b337.png',
     u'5bde325cdc5c9fb.png',
     u'a4d4967273292d2.png',
     u'23b08d245124d3c.png',
     u'a535502c45b16f6.png',
     u'8b27d32b2738fce.png',
     u'62d52e5875f15f2.png',
     u'acc6b030ec1db54.png',
     u'db4e9e9fba352e8.png',
     u'93cdbab1859dd05.png',
     u'dc311ef87140544.png',
     u'be5020af1c11fb0.png',
     u'831233abfc981bb.png',
     u'f8cbaf91c3c404f.png',
     u'c6d77ca7ad58ced.png',
     u'ca7098dc8853675.png',
     u'bde00b1efb71c8f.png',
     u'1637deef28fa753.png',
     u'e75a0c252c98431.png',
     u'05a32153f52b845.png']

    p.map(do, set(I2L_NOPOOL_50_MATCHED+I2L_STRIPS_50_MATCHED))
