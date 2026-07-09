import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import csv
from . import util
from . import html


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = 0
        self.use_html = True
        self.win_size = 160
        self.name = opt['name']
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt['display_port'])

        if self.use_html:
            self.web_dir = os.path.abspath(opt['path']['checkpoint'])
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        checkpoint_dir = os.path.abspath(opt['path']['checkpoint'])
        util.mkdirs(checkpoint_dir)
        self.log_name = os.path.join(checkpoint_dir, 'loss_log.txt')
        self.metric_csv_path = os.path.join(checkpoint_dir, 'train_metrics.csv')
        self.metric_plot_path = os.path.join(checkpoint_dir, 'train_metric_curves.png')
        self.metric_plot_interval = 10
        self.metric_history = []
        self.metric_headers = None
        self.console_hidden_metrics = set()
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save\

    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = 0 #self.opt.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    def _append_metric_history(self, epoch, i, iters, errors, lr, mode):
        progress = epoch + (float(i) / max(float(iters), 1.0))
        record = {
            'mode': mode,
            'epoch': epoch,
            'iter': i,
            'iters_per_epoch': iters,
            'progress': progress,
            'lr': lr,
        }
        for key, value in errors.items():
            record[key] = float(value)

        self.metric_history.append(record)
        if self.metric_headers is None:
            self.metric_headers = list(record.keys())
            self._rewrite_metric_csv()
        else:
            new_headers = [key for key in record.keys() if key not in self.metric_headers]
            if new_headers:
                self.metric_headers.extend(new_headers)
                self._rewrite_metric_csv()
            else:
                with open(self.metric_csv_path, "a", newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=self.metric_headers)
                    writer.writerow(record)

    def _rewrite_metric_csv(self):
        if self.metric_headers is None:
            return
        with open(self.metric_csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.metric_headers)
            writer.writeheader()
            for record in self.metric_history:
                writer.writerow(record)

    def _save_metric_curve_figure(self, metric_keys, save_path, title):
        if not metric_keys or not self.metric_history:
            return

        x = [record['progress'] for record in self.metric_history]
        n_metrics = len(metric_keys)
        ncols = 2 if n_metrics > 1 else 1
        nrows = int(np.ceil(n_metrics / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
        axes = axes.flatten()

        for idx, key in enumerate(metric_keys):
            y = [record.get(key, np.nan) for record in self.metric_history]
            axes[idx].plot(x, y, linewidth=1.5)
            axes[idx].set_title(key)
            axes[idx].set_xlabel('epoch progress')
            axes[idx].grid(True, alpha=0.3)

        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _save_metric_curves(self):
        if not self.metric_history:
            return

        metric_keys = [
            key for key in self.metric_headers
            if key not in ('mode', 'epoch', 'iter', 'iters_per_epoch', 'progress', 'lr')
        ]
        self._save_metric_curve_figure(
            metric_keys,
            self.metric_plot_path,
            self.name + ' Training Metrics'
        )



    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, iters, errors, lr, mode):
        message = '(%s - epoch: %d | iters: %d/%d | lr: %.6f) ' % (mode, epoch, i, iters, lr)
        console_errors = {
            k: v for k, v in errors.items()
            if k not in self.console_hidden_metrics
        }
        for k, v in console_errors.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        os.makedirs(os.path.dirname(self.log_name), exist_ok=True)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        self._append_metric_history(epoch, i, iters, errors, lr, mode)
        if len(self.metric_history) == 1 or len(self.metric_history) % self.metric_plot_interval == 0:
            self._save_metric_curves()

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        # short_path = ntpath.basename(image_path[0])
        # name = os.path.splitext(short_path)[0]

        short_path = image_path.split('/')
        name = short_path[-1]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_data_plt(self, webpage, visuals, pred_gt, pred, image_path):
        image_dir = webpage.get_image_dir()
        short_path = image_path.split('/')
        name = short_path[-1]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            img = image_numpy[0].cpu().float().numpy()
            fig = plt.imshow(img[0, ...])
            fig.set_cmap('gray')
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)


        image_name = '%s_%s.png' % (name, 'pred_gt')
        save_path = os.path.join(image_dir, image_name)
        img = pred_gt.astype(float)
        fig = plt.imshow(img)
        fig.set_cmap('gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        ims.append(image_name)
        txts.append('pred_gt')
        links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_result_fig(self, img, imgName, webpage, image_path):
        image_dir = webpage.get_image_dir()
        short_path = image_path.split('/')
        name = short_path[-1]
        image_name = '%s_%s.png' % (name, imgName)
        save_path = os.path.join(image_dir, image_name)
        img = img.astype(float)
        fig = plt.imshow(img)
        fig.set_cmap('gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
