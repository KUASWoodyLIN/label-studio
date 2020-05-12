import os
import cv2
import numpy as np
from glob import glob
from collections import defaultdict
from copy import deepcopy
from pyecharts.charts import Pie
from pyecharts import options as opts


rows = [i for i in range(1, 9)]


def draw_defect_heatmap(data, static_path):
    root = os.path.join(static_path, 'spindle')
    imgs_dict = {}
    for area_key, imgs_points in data.items():
        print(glob(os.path.join(root, '*', '{}.bmp'.format(area_key)))[0])
        img = cv2.imread(glob(os.path.join(root, '*', '{}.bmp'.format(area_key)))[0])
        # Create HeatMap
        heatmap = np.zeros_like(img[:, :, 0])
        height, width = heatmap.shape
        for points in imgs_points:
            points = np.array(points) / 100 * (width, height)
            heatmap += cv2.fillConvexPoly(np.zeros_like(img[:, :, 0]), np.array(points).astype(np.int), (1, 1, 1))
        heatmapshow = None
        heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        # Add HeatMap to org img
        img_out = cv2.addWeighted(img, 0.6, heatmapshow, 0.4, 0)
        cv2.imwrite(os.path.join(root, 'output', 'output_{}.bmp'.format(area_key)), img_out)
        imgs_dict[area_key] = img_out
    # Combine imgs
    cols_imgs = defaultdict(list)
    all_imgs = []
    for key in sorted(imgs_dict.keys()):
        cols_imgs[key.split('_')[0]].append(imgs_dict[key])
    for col_key, imgs in cols_imgs.items():
        img_out = np.hstack(imgs)
        all_imgs.append(img_out)
        cv2.imwrite(os.path.join(root, 'output', 'output_{}_ALL.bmp'.format(col_key)), img_out)
    cv2.imwrite(os.path.join(root, 'output', 'output_ALL_ALL.bmp'), np.vstack(all_imgs))



def pie_area2classes(data):
    # TODO: Remove fake data
    pie = Pie(init_opts=opts.InitOpts(width="2400px", height="1200px"))
    root = os.path.join('static', 'spindle')
    for area, defects in data.items():
        y, x = area.split('_')
        if 'A' in y:
            cols = list(set([filename.split('_')[0] for filename in sorted(os.listdir(os.path.join(root, 'C8C80')))]))
        else:
            cols = list(set([filename.split('_')[0] for filename in sorted(os.listdir(os.path.join(root, 'C9C78')))]))
        pie.add(
            series_name=area,
            data_pair=[[k, v] for k, v in defects.items()],
            center=[
                "{:.2f}%".format(90 / len(rows) * rows.index(int(x)) + 10),
                "{:.2f}%".format(90 / len(cols) * cols.index(y) + 10),
            ],
            radius=[50, 80],
            label_opts=opts.LabelOpts(False),
        )
    pie.set_global_opts(
        title_opts=opts.TitleOpts(title="Area -> Classes 各區域類別分佈"),
        legend_opts=opts.LegendOpts(
            type_="scroll",
            pos_top="5%",  # Label y軸位置
            pos_left="left",  # Label x軸位置
            orient="vertical"  # Label 垂直顯示
        ),
    )
    return pie.render_embed()


def pie_classes2area(data):
    inner_x_data = []
    inner_y_data = []
    middle_x_data = []
    middle_y_data = []
    outer_x_data = []
    outer_y_data = []
    pre_super_area = None
    for c, areas_item in data.items():
        if areas_item:
            inner_x_data.append(c)
            inner_y_data.append(areas_item.pop('total'))
            for area, num in areas_item.items():
                spuer_area = area.split('_')[0]
                if pre_super_area != spuer_area:
                    middle_x_data.append(spuer_area)
                    middle_y_data.append(num)
                    pre_super_area = spuer_area
                else:
                    middle_y_data[-1] += num
                outer_x_data.append(area)
                outer_y_data.append(num)
    inner_data_pair = [list(z) for z in zip(inner_x_data, inner_y_data)]
    middle_data_pair = [list(z) for z in zip(middle_x_data, middle_y_data)]
    outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]

    pie = (
        Pie(init_opts=opts.InitOpts(width="1500px", height="1050px"))
        .add(
            series_name="瑕疵類別",
            data_pair=inner_data_pair,
            radius=[0, "30%"],
            label_opts=opts.LabelOpts(
                position="outside",
            ),
        )
            .add(
            series_name="區域瑕疵數",
            radius=["45%", "60%"],
            data_pair=middle_data_pair,
            label_opts=opts.LabelOpts(
                position="outside",
                formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
                # formatter="   {b|{b}: }{c}  {per|{d}%}   ",
                background_color="#eee",
                border_color="#aaa",
                border_width=1,
                border_radius=4,
                rich={
                    "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                    "abg": {
                        "backgroundColor": "#e3e3e3",
                        "width": "100%",
                        "align": "right",
                        "height": 22,
                        "borderRadius": [4, 4, 0, 0],
                    },
                    "hr": {
                        "borderColor": "#aaa",
                        "width": "100%",
                        "borderWidth": 0.5,
                        "height": 0,
                    },
                    "b": {"fontSize": 16, "lineHeight": 33},
                    "per": {
                        "color": "#eee",
                        "backgroundColor": "#334455",
                        "padding": [2, 4],
                        "borderRadius": 2,
                    },
                },
            ),
        )
        .add(
            series_name="區域瑕疵數",
            radius=["75%", "90%"],
            data_pair=outer_data_pair,
            label_opts=opts.LabelOpts(
                position="outside",
                formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
                # formatter="   {b|{b}: }{c}  {per|{d}%}   ",
                background_color="#eee",
                border_color="#aaa",
                border_width=1,
                border_radius=4,
                rich={
                    "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                    "abg": {
                        "backgroundColor": "#e3e3e3",
                        "width": "100%",
                        "align": "right",
                        "height": 22,
                        "borderRadius": [4, 4, 0, 0],
                    },
                    "hr": {
                        "borderColor": "#aaa",
                        "width": "100%",
                        "borderWidth": 0.5,
                        "height": 0,
                    },
                    "b": {"fontSize": 16, "lineHeight": 33},
                    "per": {
                        "color": "#eee",
                        "backgroundColor": "#334455",
                        "padding": [2, 4],
                        "borderRadius": 2,
                    },
                },
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Classes -> Area 各類別區分佈"),
            legend_opts=opts.LegendOpts(pos_top="5%", pos_left="left", orient="vertical")
        )
        .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
            )
        )
    )
    return pie.render_embed()
