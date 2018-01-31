#!/usr/bin/env python2
import os, sys
import argparse as arg
commons_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src/commons'))
sys.path.append(commons_path)
import pub_commons as pub

parser = arg.ArgumentParser(description='Builds markdown content listing filenames passed in')
parser.add_argument("outname", type=str,
                    help="Output file path")
# parser.add_argument("filenames", nargs="*",
#                     help="Space separated list of filenames to include in the index")
parser.add_argument("-m", dest="modelname", type=str, default=None,
                    help="Name of the model. Will get embedded into the page title and content. If absent, the script will try to infer it.")

args = parser.parse_args()
# filenames = args.filenames
outname = args.outname
modelname = args.modelname or 'I2L-STRIPS' if 'I2L-STRIPS' in os.path.abspath(outname) else 'I2L-NOPOOL' if 'I2L-NOPOOL' in os.path.abspath(outname) else None


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

filenames =  set(I2L_STRIPS_50_MATCHED) & set(I2L_NOPOOL_50_MATCHED)
print('Num Files=%d'%len(filenames))

def gen_model_index(modelname):
    lines = []
    lines.append('---\ntitle: %s Attentive Scan\n---\n### Examples of Attentive Scan (%s)\n' % (modelname, modelname))
    lines.append('\n')
    lines.append('|Sample|Image|\n')
    lines.append('|-----|-----|\n')

    alpha_basepath = 'https://storage.googleapis.com/i2l/%s/alpha' % modelname
    image_basepath = 'https://storage.googleapis.com/i2l/data/dataset5/formula_images'

    for f in filenames:
        # sample = pub.rmheads(pub.rmtails(f, '.html', '_gray'), 'alpha_')
        sample = pub.rmtails(f, '.png')
        image = '%s_basic.png'%sample
        line = '|[%s](%s/alpha_%s_gray.html)|![](%s/%s)|\n'%(sample, alpha_basepath, sample, image_basepath, image)
        lines.append(line)
        print(line)
    return lines

def gen_main_index():
    lines = []
    lines.append('---\ntitle: Attentive Scan\n---\n')
    lines.append('### Examples of Attentive Scan  \n')
    lines.append('(Click on the model name of the sample you want to see)  \n')
    lines.append('\n')
    lines.append('|Model|Sample|\n')
    lines.append('|:----|:---:|\n')

    alpha_basepath = 'https://storage.googleapis.com/i2l/%s/alpha'
    image_basepath = 'https://storage.googleapis.com/i2l/data/dataset5/formula_images'
    def get_link(model, sample):
        return '<span style="white-space: nowrap"><a href="https://storage.googleapis.com/i2l/%s/alpha/alpha_%s_gray.html">%s</a><span/>' % (model, sample, model)
        # return ('[%s](' + alpha_basepath + '/alpha_%s_gray.html)') % (model, model, sample)

    for f in filenames:
        # sample = pub.rmheads(pub.rmtails(f, '.html', '_gray'), 'alpha_')
        sample = pub.rmtails(f, '.png')
        image = '%s_basic.png'%sample
        # line = '|[%s"](%s/alpha_%s_gray.html)|![](%s/%s)|\n'%(sample, alpha_basepath, sample, image_basepath, image)
        line = '|%s<br>%s|![](%s/%s)|\n' % (get_link(r'I2L-NOPOOL', sample), get_link(r'I2L-STRIPS', sample), image_basepath, image)
        lines.append(line)
        print(line)
    return lines

if modelname is None:
    lines = gen_main_index()
else:
    lines = gen_model_index(modelname)

with open(args.outname, 'w') as f:
    f.writelines(lines)
    f.write('\n')

