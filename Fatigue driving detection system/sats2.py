
# import math
from pyecharts.globals import ThemeType
# from pyecharts.charts import Bar, Line, Scatter
# from pyecharts import options as opts
# from pyecharts.charts import Line3D
# from pyecharts.faker import Faker
from pyecharts.charts import Bar, Liquid, Page, Pie
from pyecharts import options as opts
# pyinstaller -F run.py --distpath . --add-data /usr/local/lib/python3.8/site-packages/pyecharts/datasets:pyecharts/datasets
# --add-data=/usr/local/lib/python3.8/site-packages/pyecharts/render/templates:pyecharts/render/templates



A = ['清醒状态', '轻度疲劳', '中度疲劳', '重度疲劳']
B = [42, 21, 30, 7]
def bar6() ->Liquid:
    c = (
        Liquid()
            .add("lq", [0.65], is_outline_show=False)
            .set_global_opts(title_opts=opts.TitleOpts(title="今日湿度"))
    )
    return c


def bar7() ->Pie:
    c = (
        Pie()
        .add(
            "",
            [list(z) for z in zip(A, B)],
            radius=["40%", "75%"],
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="驾驶员疲劳分布图"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%"),
        )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    )
    return c
def bar1() ->Bar:
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
        .add_xaxis(
            [
                "0-5(秒)",
                "5-10(秒)",
                "10-15(秒)",
                "15-20(秒)",
                "25-30(秒)",
                "35-40(秒)",
            ]
        )
        .add_yaxis("驾驶员疲劳程度", [10, 20, 30, 60, 50, 80])
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            title_opts=opts.TitleOpts(title="驾驶员疲劳程度实时变化图", pos_left=20),
            datazoom_opts=opts.DataZoomOpts(is_show=True, is_realtime=True),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            legend_opts=opts.LegendOpts(pos_right='20%'),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True),effect_opts=opts.EffectOpts(is_show=True),
                         markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_='max', name='max'),
                                                                opts.MarkLineItem(type_='min', name='min')]))
    )
    return bar
def bar2() ->Bar:
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
        .add_xaxis(
            [
                "0-5(秒)",
                "5-10(秒)",
                "10-15(秒)",
                "15-20(秒)",
                "25-30(秒)",
                "35-40(秒)",
            ]
        )
        .add_yaxis("驾驶员眨眼频率", [0.89, 0.56, 1.23, 0.56, 1.43, 0.45])
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            title_opts=opts.TitleOpts(title="驾驶员眨眼频率实时变化图", pos_left=20),
            datazoom_opts=opts.DataZoomOpts(is_show=True, is_realtime=True),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            legend_opts=opts.LegendOpts(pos_right='20%'),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True),effect_opts=opts.EffectOpts(is_show=True),
                         markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_='max', name='max'),
                                                                opts.MarkLineItem(type_='min', name='min')]))
    )
    return bar
def bar3() ->Bar:
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
        .add_xaxis(
            [
                "0-5(秒)",
                "5-10(秒)",
                "10-15(秒)",
                "15-20(秒)",
                "25-30(秒)",
                "35-40(秒)",
            ]
        )
        .add_yaxis("驾驶员打哈欠频率", [0.45, 0.73, 0.28, 0.36, 0.43, 0.12])
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            title_opts=opts.TitleOpts(title="驾驶员打哈欠频率实时变化图", pos_left=20),
            datazoom_opts=opts.DataZoomOpts(is_show=True, is_realtime=True),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            legend_opts=opts.LegendOpts(pos_right='20%'),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True),effect_opts=opts.EffectOpts(is_show=True),
                         markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_='max', name='max'),
                                                                opts.MarkLineItem(type_='min', name='min')]))
    )
    return bar
def bar4() ->Bar:
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
        .add_xaxis(
            [
                "0-5(秒)",
                "5-10(秒)",
                "10-15(秒)",
                "15-20(秒)",
                "25-30(秒)",
                "35-40(秒)",
            ]
        )
        .add_yaxis("驾驶员瞌睡点头频率", [0.45, 0.73, 0.28, 0.36, 0.43, 0.12])
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            title_opts=opts.TitleOpts(title="驾驶员瞌睡点头频率实时变化图", pos_left=20),
            datazoom_opts=opts.DataZoomOpts(is_show=True, is_realtime=True),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            legend_opts=opts.LegendOpts(pos_right='20%'),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True),effect_opts=opts.EffectOpts(is_show=True),
                         markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_='max', name='max'),
                                                                opts.MarkLineItem(type_='min', name='min')]))
    )
    return bar
def page_draggable_layout():
    page = Page(layout=Page.SimplePageLayout)
    page.add(
        bar6(),
        #bar5(),
        bar7(),
        bar1(),
        bar2(),
        bar3(),
        bar4()
    )
    page.render("fatigue_detect.html")

# if __name__ == "__main__":
#     page_draggable_layout()