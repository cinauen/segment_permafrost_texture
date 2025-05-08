'''
Interactive plotting of training metrics in jupyter notebooks

'''
import numpy as np
import holoviews as hv
import param
import matplotlib
import matplotlib.pyplot as plt
from panel.viewable import Viewer
import panel as pn


class MetricsExplorerJacc(param.Parameterized):
    '''
    this can be used to plot several plot types such as:
    -- Boxplot --
    plot_pointer = hv.BoxWhisker
    plot_params = dict(box_fill_color=dim('phase').str(), box_alpha=0.5)
    plot_pointer = for hv.Violin
    plot_params = dict(violin_color=dim('phase').str(), box_alpha=0.5)
    plot_pointer = for hv.Bars
    plot_params = dict(alpha=0.5)
    '''
    site = param.ListSelector(default=[], objects=[], precedence=3)
    # precendence is priority with which the callback is executed
    # lower values are executed earlier
    #metric_plot = param.Selector(default='acc_all', objects=[], precedence=4)
    settings_query = param.String(default='all', precedence=1)
    update = param.Boolean(default=True, precedence=2)
    #save_plot = param.Boolean(default=False, precedence=2)

    def __init__(self, metrics_df_inp, sites_lst, col_map,
                 plot_pointer, plot_params, param_merge, param_train,
                 col_name, width, height, path_out, **kwargs):
        super().__init__(**kwargs)
        '''
        colname is dict per with site as key and colornam as value (used for title since no legend....)

        '''
        self.metrics_df_inp = metrics_df_inp
        self.metrics_df_plot = metrics_df_inp
        self.col_map = col_map
        self.plot_pointer = plot_pointer
        self.plot_params = plot_params
        self.width = width
        self.height = height
        self.col_name = col_name
        self.path_out = path_out

        self.param.site.objects = sites_lst  # set choice
        self.site = [sites_lst[0]] # set default

        self.param_merge = param_merge
        self.param_train = param_train

        self.TOOLS = ['crosshair', 'pan', 'wheel_zoom', 'box_zoom',
                 'reset', 'hover', 'save']

    @param.depends('update')
    def query_metrics(self):
        if self.settings_query == 'all':
            self.metrics_df_plot = self.metrics_df_inp
        else:
            t_search = self.settings_query.split(',')
            t_search = [x.strip().replace('_', '\n') for x in t_search]
            tt_text = ['settings.str.contains("' + x + '")'
                       for x in t_search]
            self.metrics_df_plot = self.metrics_df_inp.query(
                ' or '.join(tt_text))

        unique_sets = self.metrics_df_plot.index.to_frame(index=False)[['folder', 'merge_id', 'train_id']].drop_duplicates()
        self.settings_type = unique_sets['folder'].tolist()
        self.merge_id = unique_sets['merge_id'].tolist()
        self.train_id = unique_sets['train_id'].tolist()
        #self.n_settings_type = len(self.settings_type)

        #self.settings_type = self.metrics_df_plot.index.get_level_values(
        #    level='settings').unique().tolist()
        #self.merge_id = list(
        #    set([x.split('\n')[0] for x in self.settings_type]))
        #self.train_id = list(
        #    set(['_'.join(x.split('\n')[1:]).replace('\n', '_') for x in self.settings_type]))
        #print(self.train_id)

    @param.depends('settings_query')
    def table(self):
        self.param_merge['sel'] = 1
        self.param_merge.loc[self.merge_id, 'sel'] = 0

        self.param_train['sel'] = 1
        self.param_train.loc[self.train_id, 'sel'] = 0

        return hv.NdLayout(
            {'merge setttigns': hv.Table(self.param_merge.reset_index(),
                                         kdims='PARAM_PREP_ID').opts(width=int(self.width/2), height=200),
             'train setttigns': hv.Table(self.param_train.reset_index(),
                                         kdims='PARAM_TRAIN_ID').opts(width=int(self.width/2), height=200)})

    @param.depends('site')
    def load_plot(self):
        self.layout = []
        title_inp = []
        for e, i in enumerate(self.site):
            plot_inp = self.metrics_df_plot[i].reset_index()
            plot = self.plot_pointer(
                plot_inp, ['metrics', 'settings', 'phase'],  'val',
                label=i)
            plot.opts(cmap=self.col_map[i], ylim=(0, 1), **self.plot_params)
            self.layout.append(plot)

            title_inp.append(f'{i}: {self.col_name[i]}')

        self.title = ' | '.join(title_inp)
        return hv.Overlay(self.layout).opts(show_legend=True, title=self.title)


class AdvancedMetricsExplorerJacc(Viewer, MetricsExplorerJacc):

    def __init__(self, metrics_df_inp, sites_lst, col_map,
                 plot_pointer, plot_param, param_merge, param_train, col_name, width=1800,
                 height=500, path_out=None, fontsize_dict=None):
        super().__init__(metrics_df_inp, sites_lst,
                         col_map, plot_pointer, plot_param,
                         param_merge, param_train, col_name, width, height, path_out)
        self.width = width
        self.height = height
        self.fontsize_dict = fontsize_dict

    def view(self):
        self.query_metrics()

        temp = hv.DynamicMap(self.load_plot)
        if isinstance(self.fontsize_dict, str):
            self.fontsize_dict={
                'title': 16,
                'labels': 16,
                'xticks': 14,
                'yticks': 14,
                }
            opts_add = dict(fontsize=self.fontsize_dict)
        else:
            opts_add = dict()


        return temp.opts(width=self.width, height=self.height, show_grid=True,
                         legend_position='right', framewise=True,
                         **opts_add)

    def __panel__(self):
        return pn.Column(
            pn.Param(self, name="Settings",
                     widgets={#'metric_plot': pn.widgets.RadioBoxGroup,
                              'settings_query': pn.widgets.TextInput,
                              'site': {'type': pn.widgets.MultiSelect, "width": 300}
                              },
                              default_layout=pn.Row, width=1000, expand=True),
            pn.Column(
                self.view,
                self.table
                ),
            width=self.width)


class MetricsExplorer(param.Parameterized):
    '''
    this can be used to plot several plot types such as:
    -- Boxplot --
    plot_pointer = hv.BoxWhisker
    plot_params = dict(box_fill_color=dim('phase').str(), box_alpha=0.5)
    plot_pointer = for hv.Violin
    plot_params = dict(violin_color=dim('phase').str(), box_alpha=0.5)
    plot_pointer = for hv.Bars
    plot_params = dict(alpha=0.5)
    '''
    site = param.ListSelector(default=[], objects=[], precedence=3)
    # precendence is priority with which the callback is executed
    # lower values are executed earlier
    metric_plot = param.Selector(default='jacc_macro', objects=[], precedence=4)
    settings_query = param.String(default='all', precedence=1)
    update = param.Boolean(default=True, precedence=2)

    def __init__(self, metrics_df_inp, sites_lst, metrics_lst, col_map,
                 plot_pointer, plot_params, param_merge, param_train,
                 col_name, width, **kwargs):
        super().__init__(**kwargs)
        '''
        colname is dict per with site as key and colornam as value (used for title since no legend....)

        '''
        self.metrics_df_inp = metrics_df_inp
        self.metrics_df_plot = metrics_df_inp
        self.col_map = col_map
        self.plot_pointer = plot_pointer
        self.plot_params = plot_params
        self.width = width
        self.col_name = col_name

        self.param.site.objects = sites_lst  # set choice
        self.site = [sites_lst[0]] # set default
        self.param.metric_plot.objects = metrics_lst

        self.param_merge = param_merge
        self.param_train = param_train

        self.TOOLS = ['crosshair', 'pan', 'wheel_zoom', 'box_zoom',
                 'reset', 'hover', 'save']

    @param.depends('update')
    def query_metrics(self):
        if self.settings_query == 'all':
            self.metrics_df_plot = self.metrics_df_inp
        else:
            t_search = self.settings_query.split(',')
            t_search = [x.strip().replace('_', '\n') for x in t_search]
            tt_text = ['settings.str.contains("' + x + '")'
                       for x in t_search]
            self.metrics_df_plot = self.metrics_df_inp.query(
                ' or '.join(tt_text))


        unique_sets = self.metrics_df_plot.index.to_frame(index=False)[['folder', 'merge_id', 'train_id']].drop_duplicates()
        self.settings_type = unique_sets['folder'].tolist()
        self.merge_id = unique_sets['merge_id'].tolist()
        self.train_id = unique_sets['train_id'].tolist()
        #self.n_settings_type = len(self.settings_type)

        #self.settings_type = self.metrics_df_plot.index.get_level_values(
        #    level='settings').unique().tolist()
        #self.merge_id = list(
        #    set([x.split('\n')[0] for x in self.settings_type]))
        #self.train_id = list(
        #    set(['_'.join(x.split('\n')[1:]).split('_cv')[0].replace('\n', '_') for x in self.settings_type]))
        #print(self.train_id)

    @param.depends('settings_query')
    def table(self):
        self.param_merge['sel'] = 1
        self.param_merge.loc[self.merge_id, 'sel'] = 0

        self.param_train['sel'] = 1
        self.param_train.loc[self.train_id, 'sel'] = 0

        return hv.NdLayout(
            {'merge setttigns': hv.Table(self.param_merge.reset_index(),
                                         kdims='PARAM_PREP_ID').opts(width=int(self.width/2), height=200),
             'train setttigns': hv.Table(self.param_train.reset_index(),
                                         kdims='PARAM_TRAIN_ID').opts(width=int(self.width/2), height=200)})

    @param.depends('site', 'metric_plot')
    def load_plot(self):
        layout = []
        title_inp = []
        for e, i in enumerate(self.site):
            plot_inp = self.metrics_df_plot[i].reset_index()
            plot = self.plot_pointer(
                plot_inp, ['settings', 'phase'],
                self.metric_plot, label=i)
            plot.opts(cmap=self.col_map[i], ylim=(0, 1), **self.plot_params)
            layout.append(plot)

            title_inp.append(f'{i}: {self.col_name[i]}')

        title = ' | '.join(title_inp)
        return hv.Overlay(layout).opts(show_legend=True, title=title)


class AdvancedMetricsExplorer(Viewer, MetricsExplorer):

    def __init__(self, metrics_df_inp, sites_lst, metrics_lst, col_map,
                 plot_pointer, plot_param, param_merge, param_train,
                 col_name, width=1800, fontsize_dict=None):
        super().__init__(metrics_df_inp, sites_lst, metrics_lst,
                         col_map, plot_pointer, plot_param,
                         param_merge, param_train, col_name, width)
        self.width = width
        self.fontsize_dict =fontsize_dict

    def view(self):
        self.query_metrics()

        temp = hv.DynamicMap(self.load_plot)
        if isinstance(self.fontsize_dict, str):
            # e.g. use 'auto'
            self.fontsize_dict={
                'title': 16,
                'labels': 16,
                'xticks': 14,
                'yticks': 14,
                }
            opts_add = dict(fontsize=self.fontsize_dict)
        else:
            opts_add = dict()

        return temp.opts(width=self.width, height=500, show_grid=True,
                         legend_position='right', framewise=True,
                         **opts_add)

    def __panel__(self):
        return pn.Column(
            pn.Param(self, name="Settings",
                     widgets={'metric_plot': pn.widgets.RadioBoxGroup,
                              'settings_query': pn.widgets.TextInput,
                              'site': {'type': pn.widgets.MultiSelect, "width": 300}
                              },
                              default_layout=pn.Row, width=1000, expand=True),
            pn.Column(
                self.view,
                self.table
                ),
            width=self.width)


class IoUCurveExplorer(param.Parameterized):
    '''
    '''
    site = param.ListSelector(default=[], objects=[], precedence=2)
    settings_query = param.String(default='all', precedence=1)
    class_inp = param.ListSelector(default=[], objects=[], precedence=3)
    # precendence is priority with which the callback is executed
    # lower values are executed earlier
    #metric_plot = param.Selector(default='acc_all', objects=[], precedence=4)
    #settings_query = param.String(default='all', precedence=1)
    update = param.Boolean(default=True, precedence=3)
    plot_pts = param.Boolean(default=False, precedence=3)

    def __init__(self, train_metrics_df_inp, test_metrics_df_inp,
                 sites_lst, col_map, line_dash, line_width, marker_type,
                 param_merge, param_train, width, height=500, **kwargs):
        super().__init__(**kwargs)
        self.train_metrics_df_inp = train_metrics_df_inp
        self.train_metrics_df_plot = train_metrics_df_inp
        self.classes_avail = self.train_metrics_df_plot.columns.get_level_values(level='metrics').unique().tolist()

        self.test_metrics_df_inp = test_metrics_df_inp
        self.test_metrics_df_plot = test_metrics_df_inp

        self.width = width
        self.height = height

        self.col_map = col_map

        self.param.site.objects = sites_lst  # set choice
        self.site = [sites_lst[0]] # set default
        self.param.class_inp.objects = self.classes_avail
        self.class_inp = [self.classes_avail[0]] # set default

        self.line_dash = line_dash
        self.line_width = line_width
        self.marker_type = marker_type

        self.TOOLS = ['crosshair', 'pan', 'wheel_zoom', 'box_zoom',
                      'reset', 'hover', 'save']

        self.param_merge = param_merge
        self.param_train = param_train

    @param.depends('update', 'settings_query', 'site')
    def query_metrics(self):
        if self.settings_query == 'all':
            self.train_metrics_df_plot = self.train_metrics_df_inp.loc[self.site, :]
            if self.test_metrics_df_inp is not None:
                self.test_metrics_df_plot = self.test_metrics_df_inp[self.site, :]
            else:
                self.test_metrics_df_plot = self.test_metrics_df_inp
        else:
            t_search = self.settings_query.split(',')
            t_search = [x.strip() for x in t_search]
            tt_text = ['folder.str.contains("' + x + '")'
                       for x in t_search]
            self.train_metrics_df_plot = self.train_metrics_df_inp.loc[self.site, :].query(
                ' or '.join(tt_text))
            if self.test_metrics_df_inp is not None:
                self.test_metrics_df_plot = self.test_metrics_df_inp.loc[self.site, :].query(
                    ' or '.join(tt_text))
            else:
                self.test_metrics_df_plot = self.test_metrics_df_inp

        #self.train_metrics_df_plot = self.train_metrics_df_plot.loc[self.site, :]
        #if self.test_metrics_df_inp is not None:
        #    self.test_metrics_df_plot = self.test_metrics_df_plot.loc[self.site, :]

        unique_sets = self.train_metrics_df_plot.index.to_frame(index=False)[['folder', 'merge_id', 'train_id']].drop_duplicates()
        self.settings_type = unique_sets['folder'].tolist()
        self.merge_id = unique_sets['merge_id'].tolist()
        self.train_id = unique_sets['train_id'].tolist()
        self.n_settings_type = len(self.settings_type)

        #self.merge_id = list(
        #    np.unique([x.split('_')[0] for x in self.settings_type]))
        #self.train_id = list(
        #    np.unique(['_'.join(x.split('_')[1:]).split('_cv')[0] for x in self.settings_type]))

        self.color_dict = {
                i_class: cmap_to_hex(
                    self.col_map[i_class], num_bin=self.n_settings_type + 3)[3:]
                for i_class in self.classes_avail}

        self.load_plot()
        self.table()
        #self.update = True


    @param.depends('update')
    def table(self):
        self.param_merge['sel'] = 1
        self.param_merge.loc[self.merge_id, 'sel'] = 0

        self.param_train['sel'] = 1
        self.param_train.loc[self.train_id, 'sel'] = 0

        return hv.NdLayout(
            {'merge settings': hv.Table(self.param_merge.reset_index(),
                                         kdims='PARAM_PREP_ID').opts(width=int(self.width/2), height=200),
             'train settings': hv.Table(self.param_train.reset_index(),
                                         kdims='PARAM_TRAIN_ID').opts(width=int(self.width/2), height=200)})


    @param.depends('update', 'class_inp', 'plot_pts')
    def load_plot(self):

        plot_train = []
        plot_validate = []
        count = 0
        for i_site in self.site:
            for i_class in self.class_inp:
                df_plot1 = self.train_metrics_df_plot.loc[i_site, i_class]
                if self.test_metrics_df_inp is not None:
                    df_plot_test1 = self.test_metrics_df_plot.loc[i_site, i_class]
                # it is required to get settings here again since not all sites
                # have same settings
                #settings_t = df_plot1.index.get_level_values(level='folder').unique().tolist()
                unique_sets = df_plot1.index.to_frame(index=False)[
                    ['folder', 'merge_id', 'train_id']].drop_duplicates()
                settings_t = unique_sets['folder'].tolist()
                merge_id_t = unique_sets['merge_id'].tolist()
                train_id_t = unique_sets['train_id'].tolist()
                
                for e, i_settings in enumerate(settings_t):
                    df_plot = df_plot1.loc[i_settings, :].reset_index()
                    if self.test_metrics_df_inp is not None:
                        df_plot_test = df_plot_test1.loc[i_settings, :].reset_index()

                    label_txt = (
                        f"{i_site}_{'_'.join(i_class.split('_')[2:])}_{merge_id_t[e]}{train_id_t[e]}")

                    plot_train.append(hv.Curve(
                            df_plot, ('epoch', 'epoch'), ('train', 'train IoU'),
                            label=label_txt).opts(
                                xlim=(0, 100), ylim=(0, 1), alpha=0.8,
                                color=self.color_dict[i_class][e],
                                line_dash=self.line_dash[i_site],
                                line_width=self.line_width[i_site],
                                width=int(self.width/2), height=int(self.height*0.6), tools=self.TOOLS))

                    plot_validate.append(hv.Curve(
                            df_plot, ('epoch', 'epoch'), ('validate', 'validate IoU'),
                            label=label_txt).opts(
                                xlim=(0, 100), ylim=(0, 1), alpha=0.8,
                                color=self.color_dict[i_class][e],
                                line_dash=self.line_dash[i_site],
                                line_width=self.line_width[i_site],
                                width=int(self.width/2), height=int(self.height*0.6), tools=self.TOOLS))

                    if self.plot_pts and self.test_metrics_df_inp is not None:
                        plot_validate.append(hv.Scatter(
                            df_plot_test, ('epoch', 'epoch'), ('test', 'IoU'),
                            label='test ' + label_txt).opts(
                                marker=self.marker_type[i_site],
                                size=10,
                                xlim=(0, 100), ylim=(0, 1), alpha=0.8,
                                color=self.color_dict[i_class][e],
                                width=int(self.width/2), height=int(self.height*0.6), tools=self.TOOLS,
                                show_legend=False))
                    count += 1

        return hv.NdLayout(
            {'train': hv.Overlay(plot_train).opts(
                show_grid=True, legend_position='bottom',
                legend_cols=1,  # Force legend into a single column
                legend_limit=150, title='train'),  # tools=self.TOOLS
            'validate': hv.Overlay(plot_validate).opts(
                show_grid=True, legend_position='bottom',
                legend_cols=1,  # Force legend into a single column
                legend_limit=150, title='validate')},
                kdims='phase').opts(show_title=False).cols(2)


class AdvancedIoUCurveExplorer(Viewer, IoUCurveExplorer):

    def __init__(self, train_metrics_df_inp, test_metrics_df_inp, sites_lst, col_map,
                 line_dash, line_width, marker_type, param_merge, param_train, width=1800,
                 hight=500):
        super().__init__(
            train_metrics_df_inp, test_metrics_df_inp, sites_lst, col_map,
            line_dash, line_width, marker_type, param_merge, param_train,
            width, height=hight)
        self.width = width
        self.hight = hight

    def view(self):
        self.query_metrics()

        temp = hv.DynamicMap(self.load_plot)

        return temp.opts(framewise=True, width=self.width, height=self.hight)

    def __panel__(self):
        return pn.Column(
            pn.Param(self, name="Settings",
                     widgets={'settings_query': pn.widgets.TextInput,
                              'site': {'type': pn.widgets.MultiSelect, "width": 300},
                              'class_inp': pn.widgets.MultiSelect(
                                  name='class_inp', value=[self.classes_avail[0]],
                                  options=self.param.class_inp.objects, height=200),
                             },
                     default_layout=pn.Row, width=1000, height=200, expand=True),
            pn.Column(
                self.view,
                self.table
                ),
            width=self.width, height=self.hight)


class MetricsAllExplorer(param.Parameterized):
    '''
    '''
    site = param.ListSelector(default=[], objects=[], precedence=2)
    settings_query = param.String(default='all', precedence=1)
    metrics_inp = param.ListSelector(default=[], objects=[], precedence=4)
    # precendence is priority with which the callback is executed
    # lower values are executed earlier
    #metric_plot = param.Selector(default='acc_all', objects=[], precedence=4)
    #settings_query = param.String(default='all', precedence=1)
    update = param.Boolean(default=True, precedence=3)
    plot_pts = param.Boolean(default=False, precedence=4)

    def __init__(self, train_metrics_df_inp, test_metrics_df_inp,
                 sites_lst, col_map, line_dash, line_width, marker_type,
                 param_merge, param_train, width, **kwargs):
        super().__init__(**kwargs)
        self.train_metrics_df_inp = train_metrics_df_inp
        self.train_metrics_df_plot = train_metrics_df_inp
        self.metrics_avail = self.train_metrics_df_plot.columns.get_level_values(level='metrics').unique().tolist()

        self.test_metrics_df_inp = test_metrics_df_inp
        self.test_metrics_df_plot = test_metrics_df_inp

        self.col_map = col_map  # here is list and choosed according to
        # metrics selection
        self.width = width

        self.param.site.objects = sites_lst  # set choice
        self.site = [sites_lst[0]] # set default
        self.param.metrics_inp.objects = self.metrics_avail
        self.metrics_inp = [self.metrics_avail[0]] # set default

        self.line_dash = line_dash
        self.line_width = line_width
        self.marker_type = marker_type

        self.TOOLS = ['crosshair', 'pan', 'wheel_zoom', 'box_zoom',
                      'reset', 'hover', 'save']

        self.param_merge = param_merge
        self.param_train = param_train

    @param.depends('update', 'settings_query', 'site')
    def query_metrics(self):
        if self.settings_query == 'all':
            self.train_metrics_df_plot = self.train_metrics_df_inp.loc[self.site, :]#.copy()
            if self.test_metrics_df_inp is not None:
                self.test_metrics_df_plot = self.test_metrics_df_inp.loc[self.site, :]#.copy()
            else:
                self.test_metrics_df_plot = self.test_metrics_df_inp

        else:
            t_search = self.settings_query.split(',')
            t_search = [x.strip() for x in t_search]
            tt_text = ['folder.str.contains("' + x + '")'
                       for x in t_search]
            self.train_metrics_df_plot = self.train_metrics_df_inp.loc[self.site, :].query(
                ' or '.join(tt_text))#.copy()
            if self.test_metrics_df_inp is not None:
                self.test_metrics_df_plot = self.test_metrics_df_inp.loc[self.site, :].query(
                    ' or '.join(tt_text))#.copy()
            else:
                self.test_metrics_df_plot = self.test_metrics_df_inp

        #self.train_metrics_df_plot = self.train_metrics_df_plot.loc[self.site, :]
        #if self.test_metrics_df_inp is not None:
        #    self.test_metrics_df_plot = self.test_metrics_df_plot.loc[self.site, :]

        unique_sets = self.train_metrics_df_plot.index.to_frame(index=False)[['folder', 'merge_id', 'train_id']].drop_duplicates()
        self.settings_type = unique_sets['folder'].tolist()
        self.merge_id = unique_sets['merge_id'].tolist()
        self.train_id = unique_sets['train_id'].tolist()
        self.n_settings_type = len(self.settings_type)

        #self.settings_type = self.train_metrics_df_plot.index.get_level_values(level='folder').unique().tolist()
        #self.n_settings_type = len(self.settings_type)
        #self.merge_id = list(
        #    np.unique([x.split('_')[0] for x in self.settings_type]))
        #self.train_id = list(
        #    np.unique(['_'.join(x.split('_')[1:]).split('_cv')[0] for x in self.settings_type]))

        self.color_lst = [
                cmap_to_hex(
                    x, num_bin=self.n_settings_type + 3)[3:]
                for x in self.col_map]

        self.load_plot()
        self.table()



    @param.depends('update')
    def table(self):
        self.param_merge['sel'] = 1
        self.param_merge.loc[self.merge_id, 'sel'] = 0

        self.param_train['sel'] = 1
        self.param_train.loc[self.train_id, 'sel'] = 0

        return hv.NdLayout(
            {'merge settings': hv.Table(self.param_merge.reset_index(),
                                         kdims='PARAM_PREP_ID').opts(width=int(self.width/2), height=200),
             'train settings': hv.Table(self.param_train.reset_index(),
                                         kdims='PARAM_TRAIN_ID').opts(width=int(self.width/2), height=200)})


    @param.depends('update', 'metrics_inp', 'plot_pts')
    def load_plot(self):

        plot_train = []
        plot_validate = []
        count = 0
        for e_site, i_site in enumerate(self.site):
            df_plot1 = self.train_metrics_df_plot.loc[i_site, :]
            if self.test_metrics_df_inp is not None:
                df_plot_test1 = self.test_metrics_df_plot.loc[i_site, :]
            settings_plot =  df_plot1.index.get_level_values(level='folder').unique().tolist()

            for e, i_settings in enumerate(settings_plot):
                for e_metrics, i_metrics in enumerate(self.metrics_inp):
                    df_plot = df_plot1.loc[i_settings, i_metrics].reset_index()
                    if i_metrics != 'loss' and self.test_metrics_df_inp is not None:
                        df_plot_test = df_plot_test1.loc[i_settings, i_metrics].reset_index()

                    label_txt = i_site + '_' + i_metrics + '_' + '_'.join(i_settings.split('_')[:-2])

                    plot_train.append(hv.Curve(
                            df_plot, ('epoch', 'epoch'), ('train', 'metrics'),
                            label=label_txt).opts(
                                xlim=(0, 100), ylim=(0, 1), alpha=0.8,
                                color=self.color_lst[e_metrics][e],
                                line_dash=self.line_dash[i_site],
                                line_width=self.line_width[i_site],
                                width=int(self.width/2), height=500, tools=self.TOOLS))

                    plot_validate.append(hv.Curve(
                            df_plot, ('epoch', 'epoch'), ('validate', 'metrics'),
                            label=label_txt).opts(
                                xlim=(0, 100), ylim=(0, 1), alpha=0.8,
                                color=self.color_lst[e_metrics][e],
                                line_dash=self.line_dash[i_site],
                                line_width=self.line_width[i_site],
                                width=int(self.width/2), height=500, tools=self.TOOLS))

                    if self.plot_pts and i_metrics != 'loss' and self.test_metrics_df_inp is not None:
                        plot_validate.append(hv.Scatter(
                            df_plot_test, ('epoch', 'epoch'), ('test', 'metrics'),
                            label='test ' + label_txt).opts(
                                marker=self.marker_type[i_site],
                                size=10,
                                xlim=(0, 100), ylim=(0, 1), alpha=0.8,
                                color=self.color_lst[e_metrics][e],
                                width=int(self.width/2), height=500, tools=self.TOOLS,
                                show_legend=False))
                    count += 1

        return hv.NdLayout(
            {'train': hv.Overlay(plot_train).opts(
                show_grid=True, legend_position='right', legend_limit=150, title='train'),  # tools=self.TOOLS
            'validate': hv.Overlay(plot_validate).opts(
                show_grid=True, legend_position='right', legend_limit=150, title='validate')},
                kdims='phase').opts(show_title=False).cols(2)


class AdvancedMetricsAllExplorer(Viewer, MetricsAllExplorer):

    def __init__(self, train_metrics_df_inp, test_metrics_df_inp, sites_lst, col_map,
                 line_dash, line_width, marker_type, param_merge, param_train, width=1800):
        super().__init__(train_metrics_df_inp, test_metrics_df_inp, sites_lst, col_map,
                         line_dash, line_width, marker_type, param_merge, param_train,
                         width)
        self.width = width

    def view(self):
        self.query_metrics()

        temp = hv.DynamicMap(self.load_plot)

        return temp.opts(framewise=True, width=self.width, height=500)

    def __panel__(self):
        return pn.Column(
            pn.Param(self, name="Settings",
                     widgets={'settings_query': pn.widgets.TextInput,
                              'site': {'type': pn.widgets.MultiSelect, "width": 300},
                              'metrics_inp': pn.widgets.MultiSelect(
                                  name='metrics_inp', value=self.metrics_inp,
                                  options=self.param.metrics_inp.objects, height=200)},
                     default_layout=pn.Row, width=1000, height=200, expand=True),
            pn.Column(
                self.view,
                self.table
                ),
            width=self.width)



def cmap_to_hex(cmap, num_bin=None):
    '''
    num bin can be None if use discrete colorscale (e.g. tab10) which would give out fixed colors (for tab10 ten colors)
    For continuouse xcolorscales e.g. viridis, gan use num bin to define how many output colors want
    (by default varidis is forned by 256 colors ) can use cmap_plt.N to check

    to plot cororbar to check can use:
    tt = LinearSegmentedColormap.from_list('mycmap', colors=colors, N=num_bin)
    tt
    '''
    cmap_plt = plt.get_cmap(cmap)
    if num_bin is None:
        colors = cmap_plt.colors
    else:
        colors = cmap_plt(np.linspace(0, 1, num_bin))

    return [matplotlib.colors.to_hex(x) for x in colors]

