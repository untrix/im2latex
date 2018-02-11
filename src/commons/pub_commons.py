# -*- coding: utf-8 -*-
"""
    Copyright 2017 - 2018 Sumeet S Singh

    This file is part of im2latex solution by Sumeet S Singh.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the Affero GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Affero GNU General Public License for more details.

    You should have received a copy of the Affero GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

@author: Sumeet S Singh

Works on python 2.7
"""
import os
import pandas as pd
import data_commons as dtc
from viz_commons import VisualizeStep, VisualizeDir

pd.options.display.max_rows = 150
pd.options.display.max_columns = None
pd.options.display.max_colwidth = -1
pd.options.display.width = None
pd.options.display.max_seq_items = None
pd.options.display.expand_frame_repr = True
# pd.options.display.colheader_justify = 'right'
# display.pprint_nest_depth = 1


def verbatim(s):
    s = s.strip('$')
    if r'\begin' in s:
        s = s.replace(r'\begin', r'\begIn')  # Needed fool Mathjax into not rendering the LaTeX
    return s
    # return r'\begin{verbatim}\n%s\n\end{verbatim}\n' % (s,) if r'\begin' in s else s


def get_strs(dir):
    vd = VisualizeDir(dir)
    last_step = vd.get_steps()[1][-1]
    vs = VisualizeStep(vd, 'test', last_step)
    df_strs = vs.strs( 'y', 'predicted_ids', mingle=False, trim=True, wrap_strs=True, keys=['image_name'])
    df_strs['image_name_trunc'] = df_strs.image_name.str.replace('_basic.png', '.png')
    return df_strs


def DISP_ALPHA(storedir, graph, step, normalized_dataset=True,
               sample_num=0, invert_alpha=True, words=None, gamma=1, cmap='gist_gray', image=None, show_image=True):
    dtc.makeLogger(3, set_global=True)
    # Note: Good cmap values are: gist_gray, gist_yarg, gist_heat
    # Good values of gamma_correction are 1 and 2.2/2.3
    vs = VisualizeStep(VisualizeDir(storedir, normalized_dataset=normalized_dataset), graph, step)
    df_strs = vs.strs('y', 'predicted_ids', mingle=False, trim=True, wrap_strs=True, keys=['image_name'])
    if image:
        if not image.endswith('_basic.png'):
            image = image.replace('.png', '_basic.png')
        df_strs = df_strs[df_strs.image_name.isin([image])]
        assert sample_num == 0
    else:
        df_strs = df_strs.iloc[sample_num:sample_num+1]

    vs.alpha(sample_num, invert_alpha=invert_alpha, words=words, gamma_correction=gamma,
                    cmap=cmap, index=df_strs.index, show_image=show_image)

    # df_ = pd.DataFrame(data={
    #     '$\mathbf{\hat{y}}$': [df_strs.predicted_ids.iloc[0], df_strs.predicted_ids.iloc[0].strip('$')],
    #     '$\mathbf{\hat{y}}$_len': [df_strs.predicted_ids_len.iloc[0]]*2,
    #     '$\mathbf{y}$': [df_strs.y.iloc[0], df_strs.y.iloc[0].strip('$')] ,
    #     '$\mathbf{y}$_len': [df_strs.y_len.iloc[0]]*2
    #     })


    df_ = pd.DataFrame(data={
            'length': [df_strs.y_len.iloc[0], df_strs.predicted_ids_len.iloc[0]]*2 + [''],
            'value': [df_strs.y.iloc[0], df_strs.predicted_ids.iloc[0]] +
                     [verbatim(df_strs.y.iloc[0]), verbatim(df_strs.predicted_ids.iloc[0])] +
                     [df_strs.ed.iloc[0]],
        },
        index=['$\mathbf{y}$', '$\mathbf{\hat{y}}$', '$\mathbf{y}$_seq', '$\mathbf{\hat{y}}$_seq', 'edit distance'])

    display(df_[['value', 'length']])


def rmtails(s, *tails):
    for t in tails:
        s = s.rsplit(t, 1)[0]
    return s


rmtail = rmtails


def rmheads(s, *heads):
    for h in heads:
        s = s.split(h, 1)[1]
    return s


rmhead = rmheads


def get_unmatched_images(rendered_dir, strip=False):
    with open(os.path.join(rendered_dir, 'unmatched_filenames.txt'), 'r') as f:
        unmatched = [];
        missing = []
        for fname in f:
            fname = os.path.basename(fname.strip())
            path = os.path.join(rendered_dir, 'images_pred', fname)
            if not os.path.exists(path):
                if strip:
                    missing.append(fname.rsplit('.png', 1)[0])
                else:
                    missing.append(fname)
            else:
                if strip:
                    unmatched.append(fname.rsplit('.png', 1)[0])
                else:
                    unmatched.append(fname)

    return unmatched, missing


def strip_image_name(df, col='image_name'):
    """Changes name of images from xx_basic.png to xxx.png"""
    df[col] = df[col].str.replace('_basic.png', '.png')
    return df

def disp_matched_strs(dir):
    df = pd.read_pickle(os.path.join(dir, 'gallery_data', 'df_strs_matched_100.pkl'))
    df_out = pd.DataFrame({
        'edit_distance': df.ed,
        '$\mathbf{y}$_len': df.y_len,
        '$\mathbf{y}$': df.y,
        '$\mathbf{\hat{y}}$_len': df.predicted_ids_len,
        '$\mathbf{\hat{y}}$': df.predicted_ids
    }).reset_index(drop=True)[
        ['edit_distance', '$\mathbf{y}$_len', '$\mathbf{y}$', '$\mathbf{\hat{y}}$_len', '$\mathbf{\hat{y}}$']]
    return df_out

def disp_matched_strs2(dir):
    df = pd.read_pickle(os.path.join(dir, 'gallery_data', 'df_strs_matched_100.pkl'))
    df_out = pd.DataFrame({
        'edit_distance': df.ed,
        '$\mathbf{y}$_len': df.y_len,
        '$\mathbf{y}$': df.y,
        '$\mathbf{\hat{y}}$_len': df.predicted_ids_len,
        '$\mathbf{\hat{y}}$': df.predicted_ids,
        '$\mathbf{y}$_seq': df.y.apply(verbatim, convert_dtype=False),
        '$\mathbf{\hat{y}}$_seq': df.predicted_ids.apply(verbatim, convert_dtype=False)
    }).reset_index(drop=True)[['edit_distance', '$\mathbf{y}$_len', '$\mathbf{y}$', '$\mathbf{\hat{y}}$_len',
                               '$\mathbf{\hat{y}}$', '$\mathbf{y}$_seq', '$\mathbf{\hat{y}}$_seq']]

    return df_out

def disp_unmatched(dir):
    df = pd.read_pickle(os.path.join(dir, 'gallery_data', 'unmatched_preds_sample.pkl'))
    df_out = pd.DataFrame({
        'edit_distance': df.ed,
        '$\mathbf{y}$_len': df.target_len,
        '$\mathbf{y}$': df.y,
        '$\mathbf{\hat{y}}$_len': df.pred_len,
        '$\mathbf{\hat{y}}$': df['$\hat{y}$'],
        '$\mathbf{y}$_seq': df.target_seq.apply(verbatim, convert_dtype=False),
        '$\mathbf{\hat{y}}$_seq': df.pred_seq.apply(verbatim, convert_dtype=False)
    }).reset_index(drop=True)[['edit_distance', '$\mathbf{y}$_len', '$\mathbf{y}$', '$\mathbf{\hat{y}}$_len',
                               '$\mathbf{\hat{y}}$', '$\mathbf{y}$_seq', '$\mathbf{\hat{y}}$_seq']]

    return df_out

def disp_rand_sample(dir):
    df = pd.read_pickle(os.path.join(dir, 'gallery_data', 'rand_sample_100.pkl'))
    df_out = pd.DataFrame({
        'edit_distance': df.ed,
        '$\mathbf{y}$_len': df.y_len,
        '$\mathbf{y}$': df.y,
        '$\mathbf{\hat{y}}$_len': df.predicted_ids_len,
        '$\mathbf{\hat{y}}$': df.predicted_ids,
        '$\mathbf{y}$_seq': df.y.apply(verbatim, convert_dtype=False),
        '$\mathbf{\hat{y}}$_seq': df.predicted_ids.apply(verbatim, convert_dtype=False)
    }).reset_index(drop=True)[['edit_distance', '$\mathbf{y}$_len', '$\mathbf{y}$', '$\mathbf{\hat{y}}$_len',
                               '$\mathbf{\hat{y}}$', '$\mathbf{y}$_seq', '$\mathbf{\hat{y}}$_seq']]

    return df_out

