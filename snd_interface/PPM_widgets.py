from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from PyQt5.uic import loadUiType
import numpy as np
from datetime import datetime
from matplotlib import cm
from PyQt5.QtGui import QPen
import json
import warnings


Ui_LineoutImage, QLineoutImage = loadUiType('LineoutImage.ui')
Ui_Crosshair, QCrosshair = loadUiType('Crosshair.ui')
Ui_LevelsWidget, QLevelsWidget = loadUiType('LevelsWidget.ui')
Ui_AverageWidget, QAverageWidget = loadUiType('AverageWidget.ui')
Ui_Plot, QPlot = loadUiType('Epics_plot.ui')
Ui_Config, QConfig = loadUiType('Config.ui')
Ui_Imager, QImager = loadUiType('Imager_controls.ui')
Ui_ImagerStats, QImagerStats = loadUiType('Imager_stats.ui')
Ui_wfsStats, QwfsStats = loadUiType('wfs_stats.ui')
Ui_wfs, Qwfs = loadUiType('wfs_controls.ui')
Ui_DisplayOptions, QDisplayOptions = loadUiType('Wavefront_Display_Options.ui')


class WFSDisplay(QDisplayOptions, Ui_DisplayOptions):
    """
    Widget class for choosing wavefront display
    """
    def __init__(self, parent=None):
        super(WFSDisplay, self).__init__()
        self.setupUi(self)

        # connect callbacks
        self.FOVLineEdit.returnPressed.connect(self.change_FOV)
        self.displayComboBox.currentIndexChanged.connect(self.change_display)
        self.zLineEdit.returnPressed.connect(self.change_z)
        self.imrotLineEdit.returnPressed.connect(self.change_rotation)

        # initialize attributes
        self.FOV = 0
        self.display_choice = ''
        self.focus_z = 0
        self.rotation = 0

        # call all callbacks to set default values
        self.change_FOV()
        self.change_display(0)
        self.change_z()
        self.change_rotation()

    def change_z(self):
        """
        Method to adjust the viewing plane of the focus based on user input
        """
        try:
            self.focus_z = float(self.zLineEdit.text())
        except ValueError:
            self.focus_z = 0.0
            self.zLineEdit.setText('0.0')

    def change_rotation(self):
        """
        Method to adjust the rotation of the image in case the image isn't lined up with the Talbot pattern
        """
        try:
            self.rotation = float(self.imrotLineEdit.text())
        except ValueError:
            self.rotation = 0.0
            self.imrotLineEdit.setText('0.0')

    def change_FOV(self):
        """
        Method to change the field of view of the focus (interpolated)
        """
        try:
            self.FOV = float(self.FOVLineEdit.text())
        except ValueError:
            self.FOV = 10.0
            self.FOVLineEdit.setText('10.0')

    def change_display(self, index):
        """
        Method to change the 2D display on the wavefront tab
        Parameters
        ----------
        index: int
            not used
        """
        self.display_choice = self.displayComboBox.currentText()


class WFSControls(Qwfs, Ui_wfs):
    """
    Widget class to store some imager controls
    """
    def __init__(self, parent=None):
        super(WFSControls, self).__init__()
        self.setupUi(self)

        self.wfs_prefix = self.yStateReadback.channel[5:15]

        self.widgets = [self.yStateReadback, self.yStateComboBox,
                self.yPosReadback, self.yPosLineEdit, self.zPosReadback,
                self.zPosLineEdit]

    def change_wfs(self, wfs_name):
       
        if wfs_name is None:
            for widget in self.widgets:
                self.change_channel(widget, '')
        else:
            self.wfs_prefix = wfs_name+':WFS:'
            self.change_channel(self.yStateReadback, 'MMS:STATE:GET_RBV')
            self.change_channel(self.yStateComboBox, 'MMS:STATE:SET')
            self.change_channel(self.yPosReadback, 'MMS:Y.RBV')
            self.change_channel(self.yPosLineEdit, 'MMS:Y.VAL')
            self.change_channel(self.zPosReadback, 'MMS:Z.RBV')
            self.change_channel(self.zPosLineEdit, 'MMS:Z.VAL')

    def change_channel(self, obj, suffix):

        obj.channel = 'ca://' + self.wfs_prefix + suffix


class ImagerControls(QImager, Ui_Imager):
    """
    Widget class to store some imager controls
    """
    def __init__(self, parent=None):
        super(ImagerControls, self).__init__()
        self.setupUi(self)

        self.imager_prefix = self.yStateReadback.channel[5:16]

        self.nominalButton.clicked.connect(self.restore_nominal)

        self.nominal_controls = {
                'zoom': self.zoomLineEdit,
                'focus': self.focusLineEdit,
                'ND': self.ndStateComboBox,
                'AcquireTime': self.acquireLineEdit
                }

    def restore_nominal(self):

        # read nominal imager settings
        imager_name = self.imager_prefix[0:5]


        try:
            #with open('/reg/neh/home/seaberg/Commissioning_Tools/PPM_centroid/imagers.db') as json_file:
            with open('imagers.db') as json_file:
                data = json.load(json_file)

        except json.decoder.JSONDecodeError:
            data = {}
            print('failed to load file')

        if imager_name in data:
            settings = data[imager_name]
            print('found imager settings')
        else:
            settings = {}
            print('imager settings not found')

        for key in settings:
            if key in self.nominal_controls:
                if isinstance(self.nominal_controls[key], QtWidgets.QLineEdit):
                    self.nominal_controls[key].setText(str(settings[key]))
                    self.nominal_controls[key].send_value()
                elif isinstance(self.nominal_controls[key], QtWidgets.QComboBox):
                    index = self.nominal_controls[key].findText(str(settings[key]))
                    self.nominal_controls[key].setCurrentIndex(index)
                    self.nominal_controls[key].internal_combo_box_activated_int(index)
                print(key+' set')

        #self.acquireLineEdit.setText('.008')
        #self.acquireLineEdit.send_value()

    def change_imager(self, imager_prefix):
        
        self.imager_prefix = imager_prefix
        self.change_channel(self.yStateReadback, 'MMS:STATE:GET_RBV')
        self.change_channel(self.yStateComboBox, 'MMS:STATE:SET')
        self.change_channel(self.zoomReadback, 'CLZ.RBV')
        self.change_channel(self.zoomLineEdit, 'CLZ.VAL')
        self.change_channel(self.focusReadback, 'CLF.RBV')
        self.change_channel(self.focusLineEdit, 'CLF.VAL')
        self.change_channel(self.ndStateReadback, 'MFW:GET_RBV')
        self.change_channel(self.ndStateComboBox, 'MFW:SET')
        self.change_channel(self.acquireReadback, 'CAM:AcquireTime_RBV')
        self.change_channel(self.acquireLineEdit, 'CAM:AcquireTime')
        self.change_channel(self.yPosReadback, 'MMS.RBV')
        self.change_channel(self.yPosLineEdit, 'MMS.VAL')

    def change_channel(self, obj, suffix):

        obj.channel = 'ca://'+self.imager_prefix+suffix


class WFSStats(QwfsStats, Ui_wfsStats):
    """
    Widget class to display image/beam stats
    """
    def __init__(self, parent=None):
        super(WFSStats, self).__init__()
        self.setupUi(self)

        self.threshold = float(self.thresholdLineEdit.text())

        self.thresholdLineEdit.returnPressed.connect(self.update_threshold)
        self.nImagesLineEdit.returnPressed.connect(self.update_num)

        self.image_widget = None
        self.num = int(self.nImagesLineEdit.text())

    def update_num(self):
        try:
            self.num = int(self.nImagesLineEdit.text())
        except ValueError:
            self.num = 1
            self.nImagesLineEdit.setText('1')

    def get_filtered(self, data):
        # get most recent data
        recent_data = data[-self.num:]
        # filter out nan's
        mask = np.logical_not(np.isnan(recent_data))

        return recent_data[mask]

    def update_stats(self, data):

        z_x = self.get_filtered(data['z_x'])
        z_y = self.get_filtered(data['z_y'])
        rms_x = self.get_filtered(data['rms_x'])
        rms_y = self.get_filtered(data['rms_y'])

        self.xFocusLineEdit.setText('%.1f' % np.mean(z_x))
        self.yFocusLineEdit.setText('%.1f' % np.mean(z_y))
        self.xWidthLineEdit.setText('%.1f' % np.mean(rms_x))
        self.yWidthLineEdit.setText('%.1f' % np.mean(rms_y))
        self.xFocusRMSLineEdit.setText('%.2f' % np.std(z_x))
        self.yFocusRMSLineEdit.setText('%.2f' % np.std(z_y))
        self.xWidthRMSLineEdit.setText('%.2f' % np.std(rms_x))
        self.yWidthRMSLineEdit.setText('%.2f' % np.std(rms_y))

    def update_threshold(self):
        try:
            self.threshold = float(self.thresholdLineEdit.text())
        except ValueError:
            self.threshold = 0.1
            self.thresholdLineEdit.setText('0.1')

    def get_threshold(self):

        return self.threshold


class ImagerStats(QImagerStats, Ui_ImagerStats):
    """
    Widget class to display image/beam stats
    """
    def __init__(self, parent=None):
        super(ImagerStats, self).__init__()
        self.setupUi(self)

        self.threshold = float(self.thresholdLineEdit.text())

        self.thresholdLineEdit.returnPressed.connect(self.update_threshold)
        self.nImagesLineEdit.returnPressed.connect(self.update_num)

        self.image_widget = None
        self.color = QtCore.Qt.green
        self.num = int(self.nImagesLineEdit.text())

        self.thickness = 8

        self.circle = QtWidgets.QGraphicsEllipseItem(0, 0, 0, 0)
        self.circle.setPen(QtGui.QPen(self.color, self.thickness, Qt.SolidLine))

        self.ref_circle = QtWidgets.QGraphicsEllipseItem(0, 0, 0, 0)
        self.ref_circle.setPen(QtGui.QPen(QtCore.Qt.white, self.thickness, Qt.SolidLine))

    def get_filtered(self, data):
        # get most recent data
        recent_data = data[-self.num:]
        # filter out nan's
        mask = np.logical_not(np.isnan(recent_data))

        return recent_data[mask]

    def update_num(self):
        try:
            self.num = int(self.nImagesLineEdit.text())
        except ValueError:
            self.num = 1
            self.nImagesLineEdit.setText('1')

    def update_width(self):
        # get width of the bounding rect
        rect_width = self.image_widget.rect.boundingRect().width()
        # set line thickness to 1% of the viewbox width
        self.thickness = rect_width * .01
        # update lines
        self.circle.setPen(QtGui.QPen(self.color, self.thickness, Qt.SolidLine))
        self.ref_circle.setPen(QtGui.QPen(QtCore.Qt.white, self.thickness, Qt.SolidLine))


    def connect_image(self, image_widget):

        self.image_widget = image_widget

        self.showFitButton.toggled.connect(self.circle_toggled)

    def update_stats(self, data):

        cx = self.get_filtered(data['cx'])
        cy = self.get_filtered(data['cy'])
        wx = self.get_filtered(data['wx'])
        wy = self.get_filtered(data['wy'])
        intensity = self.get_filtered(data['intensity'])

        cx_mean = np.mean(cx)
        cy_mean = np.mean(cy)
        wx_mean = np.mean(wx)
        wy_mean = np.mean(wy)

        self.xCentroidLineEdit.setText('%.1f' % cx_mean)
        self.yCentroidLineEdit.setText('%.1f' % cy_mean)
        self.xWidthLineEdit.setText('%.1f' % wx_mean)
        self.yWidthLineEdit.setText('%.1f' % wy_mean)
        self.xCentroidRMSLineEdit.setText('%.2f' % np.std(cx))
        self.yCentroidRMSLineEdit.setText('%.2f' % np.std(cy))
        self.xWidthRMSLineEdit.setText('%.2f' % np.std(wx))
        self.yWidthRMSLineEdit.setText('%.2f' % np.std(wy))
        self.intensityLineEdit.setText('%.2f' % np.mean(intensity))
        self.intensityRMSLineEdit.setText('%.2f' % np.std(intensity))

        xRef = data['cx_ref']
        yRef = data['cy_ref']

        distance = np.sqrt((xRef-cx_mean)**2 + (yRef-cy_mean)**2)

        if distance < 50:
            self.color = QtCore.Qt.green
        else:
            self.color = QtCore.Qt.red

        if self.showFitButton.isChecked():

            self.circle.setRect(cx_mean-wx_mean, cy_mean-wy_mean,
                    2*wx_mean, 2*wy_mean)
            self.ref_circle.setRect(xRef-wx_mean, yRef-wy_mean,
                    2*wx_mean, 2*wy_mean)
            self.circle.setPen(QtGui.QPen(self.color, self.thickness, Qt.SolidLine))
            self.ref_circle.setPen(QtGui.QPen(QtCore.Qt.white, self.thickness, Qt.SolidLine))

    def update_threshold(self):
        try:
            self.threshold = float(self.thresholdLineEdit.text())
        except ValueError:
            self.threshold = 0.1
            self.thresholdLineEdit.setText('0.1')

    def get_threshold(self):

        return self.threshold

    def circle_toggled(self, evt):

        if evt:
            self.addCircle()
        else:
            self.removeCircle()

    def addCircle(self):

        self.image_widget.view.addItem(self.ref_circle)
        self.image_widget.view.addItem(self.circle)

    def removeCircle(self):
        self.image_widget.view.removeItem(self.circle)
        self.image_widget.view.removeItem(self.ref_circle)


class Orientation:

    def __init__(self, image):

        self.orientation = 0

    def change_orientation(self, orientation):
        self.orientation = orientation


class LineoutImage(QLineoutImage, Ui_LineoutImage):
    """
    Class to represent a widget containing an image with horizontal and vertical lineouts. Linked to LineoutImage.ui.
    """
    def __init__(self, parent=None):
        """
        Initialize the widget.
        :param parent:
        """
        super(LineoutImage, self).__init__()
        self.setupUi(self)

        # initialize levels widget attribute
        self.levels = None
        # initialize crosshair widget attribute
        self.crosshairsWidget = None

        # set default image levels
        self.minimum = 0
        self.maximum = 4096

        # add viewbox for image
        self.view = self.image_canvas.addViewBox()

        # add colormap to context menu
        colormapMenu = self.view.menu.addMenu("Colormap")
        gnuplot = colormapMenu.addAction("gnuplot")
        grayscale = colormapMenu.addAction("grayscale")
        viridis = colormapMenu.addAction("viridis")
        bwr = colormapMenu.addAction("BlueWhiteRed")
        hsv = colormapMenu.addAction("hsv")

        # connect callbacks
        gnuplot.triggered.connect(self.set_gnuplot)
        grayscale.triggered.connect(self.set_grayscale)
        viridis.triggered.connect(self.set_viridis)
        bwr.triggered.connect(self.set_bwr)
        hsv.triggered.connect(self.set_hsv)

        # define colormaps
        self.colormaps = self.define_colormaps()

        # setup viewbox and get corresponding QRect
        self.rect = self.setup_viewbox(1024)
       
        # lock aspect ratio
        self.view.setAspectLocked(True)
        # add an image
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        # set default colormap
        self.img.setLookupTable(self.colormaps['gnuplot'])

        # font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')

        # initialize lineouts
        (self.horizontalPlot,
         self.horizontalLineout,
         self.horizontalFit) = self.initialize_lineout(self.xlineout_canvas, 'horizontal')
        (self.verticalPlot,
         self.verticalLineout,
         self.verticalFit) = self.initialize_lineout(self.ylineout_canvas, 'vertical')

    def get_lut(self, name):
        cmap = cm.get_cmap(name)
        cmap._init()
        lut = (cmap._lut * 255).view(np.ndarray)[:256,:]
        lut[-1,:] = 255

        return lut

    def define_colormaps(self):

        cmap_dict = {}

        cmap_dict['grayscale'] = self.get_lut("gray")
        cmap_dict['gnuplot'] = self.get_lut("gnuplot")
        cmap_dict['viridis'] = self.get_lut("viridis")
        cmap_dict['bwr'] = self.get_lut("bwr")
        cmap_dict['hsv'] = self.get_lut("hsv")

        return cmap_dict
        
    def set_viridis(self, evt):
        self.img.setLookupTable(self.colormaps['viridis'])

    def set_hsv(self, evt):
        self.img.setLookupTable(self.colormaps['hsv'])

    def set_bwr(self, evt):
        self.img.setLookupTable(self.colormaps['bwr'])
    
    def set_grayscale(self, evt):

        self.img.setLookupTable(self.colormaps['grayscale'])

    def set_gnuplot(self, evt):
        
        self.img.setLookupTable(self.colormaps['gnuplot'])

    def connect_crosshairs(self, crosshairs):
        """
        Method to connect a CrosshairWidget.
        :param crosshairs: CrosshairWidget
        :return:
        """
        # set attribute
        self.crosshairsWidget = crosshairs

        # setup crosshairs
        self.crosshairsWidget.connect_image(self)

        self.rect.scene().sigMouseClicked.connect(self.mouseClicked)


    def mouseClicked(self, evt):
        """
        Method to define new crosshair location based on mouseclick.
        :param evt: mouse click event
            Contains scene position
        :return:
        """

        # translate scene coordinates to viewbox coordinates
        coords = self.view.mapSceneToView(evt.scenePos())

        # update crosshair
        self.crosshairsWidget.update_crosshair_coords(coords)

    def connect_levels(self, levels):
        """
        Method to connect a Levels widget for scaling the image.
        :param levels: LevelsWidget
        :return:
        """
        # set attribute
        self.levels = levels

        # set levels based on current entries
        self.set_min()
        self.set_max()
        # connect line edit return to set_min, set_max methods
        self.levels.minLineEdit.returnPressed.connect(self.set_min)
        self.levels.maxLineEdit.returnPressed.connect(self.set_max)

    def get_canvases(self):
        """
        Method to give access to GraphicsLayoutWidgets
        :return:
        """
        return self.image_canvas, self.xlineout_canvas, self.ylineout_canvas

    def update_plots(self, image_data, x, y, xprojection_data, yprojection_data, fit_x, fit_y, xlineout_data=None, ylineout_data=None):
        """
        Method to update image, lineout plots
        :param image_data: (N, M) ndarray
            array corresponding to image data to display
        :param x: (M) ndarray
            1D array defining x axis coordinates
        :param y: (N) ndarray
            1D array defining y axis coordinates
        :param xlineout_data: (M) ndarray
            1D array containing horizontal image lineout
        :param ylineout_data: (N) ndarray
            1D array containing vertical image lineout
        :param fit_x: (M) ndarray
            1D array containing gaussian fit to horizontal lineout
        :param fit_y: (N) ndarray
            1D array containing gaussian fit to vertical lineout
        :return:
        """

        if xlineout_data is None or ylineout_data is None:
            xlineout_data = xprojection_data
            ylineout_data = yprojection_data

        # check if there is an associated levels widget
        if self.levels is not None:
            # check if we're autoscaling
            if self.levels.checkBox.isChecked():
                self.minimum = np.min(image_data)
                self.maximum = np.max(image_data)
                # set text on levels widget
                self.levels.setText(self.minimum, self.maximum)
        else:
            # autoscale if there is no levels widget
            self.minimum = np.min(image_data)
            self.maximum = np.max(image_data)

        # figure out image extent based on coordinates
        x_width = np.max(x) - np.min(x)
        y_width = np.max(y) - np.min(y)

        # set image data
        self.img.setImage(image_data,
                levels=(self.minimum, self.maximum))

        # set rect size based on coordinates
        self.img.setRect(QtCore.QRectF(np.min(x),np.min(y),x_width, y_width))

        # set lineout data
        
        if self.lineoutCheckBox.isChecked():
            self.horizontalLineout.setData(x, xlineout_data)
            self.horizontalFit.setData(x, fit_x)
            self.verticalLineout.setData(ylineout_data, y)
            self.verticalFit.setData(fit_y, y)
        else:
            self.horizontalLineout.setData(x, xprojection_data)
            self.horizontalFit.setData(x, fit_x)
            self.verticalLineout.setData(yprojection_data, y)
            self.verticalFit.setData(fit_y, y)

    def set_min(self):
        """
        Method called when return is pressed on levels.minLineEdit.
        :return:
        """
        # update the minimum to the new value
        self.minimum = float(self.levels.minLineEdit.text())

    def set_max(self):
        """
        Method called when return is pressed on levels.maxLineEdit.
        :return:
        """
        # update the maximum to the new value
        self.maximum = float(self.levels.maxLineEdit.text())

    def setup_viewbox(self, width):
        """
        Helper function to set up viewbox with title
        :param width: image width in pixels (int)
        """
        # lock aspect ratio
        self.view.setAspectLocked(True)
        # update viewbox range
        self.view.setRange(QtCore.QRectF(-width/2., -width/2., width, width))
        # draw a white rectangle that is the same size as the image to show the image boundary
        rect1 = QtGui.QGraphicsRectItem(-width/2., -width/2., width, width)
        rect1.setPen(QtGui.QPen(QtCore.Qt.white, width/50., QtCore.Qt.SolidLine))
        # add the rectangle to the viewbox
        self.view.addItem(rect1)
        self.view.invertX(True)
        # return the rectangle
        return rect1
        
    def update_viewbox(self, width, height):
        """
        Helper function to adjust viewbox settings
        :param width: new width in pixels (int)
        :param height: new height in pixels (int)
        :return:
        """
        # set range to new size
        self.view.setRange(QtCore.QRectF(-width/2, -height/2, width, height))
        # update the bounding rectangle
        self.rect.setPen(QtGui.QPen(QtCore.Qt.white, width/50., QtCore.Qt.SolidLine))
        self.rect.setRect(-width/2, -height/2, width, height)

    def change_lineout_label(self, ylabel):
        """
        Method to change the "y-axis" label on the lineouts
        :param ylabel: str
            New label for lineout y-axis
        :return:
        """
        # update lineout labels
        PlotUtil.label_plot(self.horizontalPlot, u'x (\u03BCm)', ylabel)
        PlotUtil.label_plot(self.verticalPlot, ylabel, u'y (\u03BCm)')

    def initialize_lineout(self, canvas, direction):
        """
        Method to set up lineout plots.
        :param canvas: pg.GraphicsLayoutWidget
            Layout widget used for adding pyqtgraph widgets
        :param direction: str
            'horizontal' or 'vertical': direction of the lineout
        """
        # legend names
        names = ['Lineout', 'Fit']
        # line colors
        colors = ['r', 'c']

        # add plot to canvas
        if direction == 'horizontal':
            # horizontal lineout
            lineoutPlot = canvas.addPlot()
            # initialize legend and adjust position
            legend = lineoutPlot.addLegend(offset=(10,0))
            # initialize lineout plot
            lineoutData = lineoutPlot.plot(np.linspace(-1024, 1023, 100), np.zeros(100),
                                           pen=pg.mkPen(colors[0], width=2),name=names[0])
            # initialize fit plot
            lineoutFit = lineoutPlot.plot(np.linspace(-1024, 1023, 100), np.zeros(100),
                                           pen=pg.mkPen(colors[1], width=2),name=names[1])

            # add legend
            PlotUtil.setup_legend(legend)

            # set range to be normalized
            #lineoutPlot.setYRange(0, 1)

            # plot labels
            PlotUtil.label_plot(lineoutPlot, u'x (\u03BCm)', 'Intensity')
            # link axis to image
            lineoutPlot.setXLink(self.view)

            lineoutPlot.invertX(True)

        elif direction == 'vertical':
            # vertical lineout
            lineoutPlot = canvas.addPlot()
            # initialize lineout plot
            lineoutData = lineoutPlot.plot(np.zeros(100), np.linspace(-1024, 1023, 100),
                                           pen=pg.mkPen(colors[0], width=2),name=names[0])
            # initialize fit plot
            lineoutFit = lineoutPlot.plot(np.zeros(100), np.linspace(-1024, 1023, 100),
                                           pen=pg.mkPen(colors[1], width=2),name=names[1])

            # set range to be normalized
            #lineoutPlot.setXRange(0, 1)
            # plot labels
            PlotUtil.label_plot(lineoutPlot, 'Intensity', u'y (\u03BCm)')
            # link axis to image
            lineoutPlot.setYLink(self.view)
        else:
            # just to catch anything weird
            lineoutPlot = None
            lineoutData = None
            lineoutFit = None

        # return the plot widget and line plots
        return lineoutPlot, lineoutData, lineoutFit


class PlotUtil:
    """
    Utility class for PPM widgets. Contains only static methods.
    """

    labelStyle = {'color': '#FFF', 'font-size': '10pt'}
    font = QtGui.QFont()
    font.setPointSize(10)
    font.setFamily('Arial')

    @staticmethod
    def setup_legend(legend):
        """
        Method for setting legend style
        :param legend: pg.LegendItem
            legend that needs formatting
        :return:
        """

        # set style
        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        # loop through legend items
        for item in legend.items:
           for single_item in item:
               # set style
               if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                   single_item.setText(single_item.text, **legendLabelStyle)

    @staticmethod
    def label_plot(plot, xlabel, ylabel):
        """
        Helper function to set plot labels
        :param plot: pyqtgraph plot item
        :param xlabel: str
            x-axis label
        :param ylabel: str
            y-axis label
        """
        # label x-axis
        xaxis = plot.getAxis('bottom')
        PlotUtil.set_axislabel(xaxis, xlabel, 'w', 1)

        # label y-axis
        yaxis = plot.getAxis('left')
        PlotUtil.set_axislabel(yaxis, ylabel, 'w', 1)

    @staticmethod
    def set_axislabel(axis, text, color, width):
        """
        Convenience method for axis labeling
        :param axis: pyqtgraph axis item
            axis being labeled
        :param text: str
            label text
        :param color: str
            character corresponding to QtPen color
        :param width: int
            width of pen
        :return:
        """
        # set label
        axis.setLabel(text=text, **PlotUtil.labelStyle)
        # set font
        axis.tickFont = PlotUtil.font
        # set pen color and size
        axis.setPen(pg.mkPen(color, width=width))


class ImageBase:

    def __init__(self, canvas):
        self.view = canvas.addViewBox()
        self.view.setAspectLocked(True)


class ImageZoom:
    """
    Class for displaying a zoomed-in image
    """

    def __init__(self, canvas, color):

        self.view = canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 90, 90))
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        rect = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        #rect.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        rect.setPen(pg.mkPen(color, width=2))
        self.view.addItem(rect)

        self.levels = None
        self.minimum = 0
        self.maximum = 4096

    def update_image(self, image_data):

        self.img.setImage(image_data, levels=(self.minimum, self.maximum))
   
    def connect_levels(self, levels):
        """
        Method to connect a Levels widget for scaling the image.
        :param levels: LevelsWidget
        :return:
        """
        # set attribute
        self.levels = levels

        # set levels based on current entries
        self.set_min()
        self.set_max()
        # connect line edit return to set_min, set_max methods
        self.levels.minLineEdit.returnPressed.connect(self.set_min)
        self.levels.maxLineEdit.returnPressed.connect(self.set_max)
   
    def set_min(self):
        """
        Method called when return is pressed on levels.minLineEdit.
        :return:
        """
        # update the minimum to the new value
        self.minimum = float(self.levels.minLineEdit.text())

    def set_max(self):
        """
        Method called when return is pressed on levels.maxLineEdit.
        :return:
        """
        # update the maximum to the new value
        self.maximum = float(self.levels.maxLineEdit.text())


class ImageRegister:
    """
    Class for displaying image registration screen
    """

    def __init__(self, canvas):

        self.canvas = canvas

        # Full image
        self.view = self.canvas.addViewBox()

        width = 1024

        self.levels = None

        self.rect = self.setup_viewbox(1024) 

        self.view.setAspectLocked(True)
        
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        self.rect_bl = QtWidgets.QGraphicsRectItem(0,0,160,160)
        self.rect_bl.setPen(QPen(Qt.cyan, 8, Qt.SolidLine))
        self.rect_br = QtWidgets.QGraphicsRectItem(1888,0,160,160)
        self.rect_br.setPen(QPen(Qt.darkMagenta, 8, Qt.SolidLine))
        self.rect_tl = QtWidgets.QGraphicsRectItem(0,1888,160,160)
        self.rect_tl.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.rect_tr = QtWidgets.QGraphicsRectItem(1888,1888,160,160)
        self.rect_tr.setPen(QPen(Qt.green, 8, Qt.SolidLine))


        #circ1 = QtWidgets.QGraphicsEllipseItem(1024-25,1024-25,50,50)
        #circ1.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        self.crossx0 = QtWidgets.QGraphicsLineItem(1024-25,1024,1024+25,1024)
        self.crossy0 = QtWidgets.QGraphicsLineItem(1024,1024-25,1024,1024+25)
        self.crossx0.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        self.crossy0.setPen(QPen(Qt.green, 8, Qt.SolidLine))

        
        #self.circ0 = QtWidgets.QGraphicsEllipseItem(1024-25,1024-25,50,50)
        #self.circ0.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.crossx = QtWidgets.QGraphicsLineItem(1024-25,1024,1024+25,1024)
        self.crossy = QtWidgets.QGraphicsLineItem(1024,1024-25,1024,1024+25)
        self.crossx.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.crossy.setPen(QPen(Qt.red, 8, Qt.SolidLine))

        
        self.circ_tl = QtWidgets.QGraphicsRectItem(256-25,1792-25,50,50)
        self.circ_tl.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.circ_tr = QtWidgets.QGraphicsRectItem(1792-25,1792-25,50,50)
        self.circ_tr.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        self.circ_bl = QtWidgets.QGraphicsRectItem(256-25,256-25,50,50)
        self.circ_bl.setPen(QPen(Qt.cyan, 8, Qt.SolidLine))
        self.circ_br = QtWidgets.QGraphicsRectItem(1792-25,256-25,50,50)
        self.circ_br.setPen(QPen(Qt.darkMagenta, 8, Qt.SolidLine))
        self.view.addItem(self.rect_bl)
        self.view.addItem(self.rect_br)
        self.view.addItem(self.rect_tl)
        self.view.addItem(self.rect_tr)
        #self.view0.addItem(circ1)
        self.view.addItem(self.crossx0)
        self.view.addItem(self.crossy0)
        self.view.addItem(self.crossx)
        self.view.addItem(self.crossy)
        #self.view0.addItem(self.circ0)
        self.view.addItem(self.circ_tl)
        self.view.addItem(self.circ_tr)
        self.view.addItem(self.circ_bl)
        self.view.addItem(self.circ_br)



        self.pix_size_text = pg.TextItem('Pixel size: %.2f microns' % 0.0,
                color=(200,200,200), border='c', fill='b',anchor=(0,1))
        self.pix_size_text.setFont(QtGui.QFont("", 10, QtGui.QFont.Bold))
        self.pix_size_text.setPos(-width/2+width/7,-width/2+width/2048)
        self.view.addItem(self.pix_size_text)

        self.update_viewbox(1024,1024)

        self.minimum = 0
        self.maximum = 4096

    def connect_levels(self, levels):
        """
        Method to connect a Levels widget for scaling the image.
        :param levels: LevelsWidget
        :return:
        """
        # set attribute
        self.levels = levels

        # set levels based on current entries
        self.set_min()
        self.set_max()
        # connect line edit return to set_min, set_max methods
        self.levels.minLineEdit.returnPressed.connect(self.set_min)
        self.levels.maxLineEdit.returnPressed.connect(self.set_max)
   
    def set_min(self):
        """
        Method called when return is pressed on levels.minLineEdit.
        :return:
        """
        # update the minimum to the new value
        self.minimum = float(self.levels.minLineEdit.text())

    def set_max(self):
        """
        Method called when return is pressed on levels.maxLineEdit.
        :return:
        """
        # update the maximum to the new value
        self.maximum = float(self.levels.maxLineEdit.text())


    def update_image(self, image_data, pixSize, center=None, scale=None):
        if self.levels is not None:
            # check if we're autoscaling
            if self.levels.checkBox.isChecked():
                self.minimum = np.min(image_data)
                self.maximum = np.max(image_data)
                # set text on levels widget
                self.levels.setText(self.minimum, self.maximum)
        else:
            # autoscale if there is no levels widget
            self.minimum = np.min(image_data)
            self.maximum = np.max(image_data)
    
        width = self.rect.rect().width()
        height = self.rect.rect().height()
        self.img.setImage(image_data,
                levels=(self.minimum, self.maximum))

        self.img.setRect(QtCore.QRectF(-width/2, -height/2, width, height))
       
       

        if center is not None:

            center = center - width/2
            rwidth = width/45
            #self.circ1.setRect(center[0,1]-scale[0]*rwidth,center[0,0]-scale[0]*rwidth,
            #    2*rwidth*scale[0],2*rwidth*scale[0])
            #self.circ2.setRect(center[1,1]-scale[1]*rwidth,center[1,0]-scale[1]*rwidth,
            #    2*rwidth*scale[1],2*rwidth*scale[1])
            #self.circ3.setRect(center[2,1]-scale[2]*rwidth,center[2,0]-scale[2]*rwidth,
            #    2*rwidth*scale[2],2*rwidth*scale[2])
            #self.circ4.setRect(center[3,1]-scale[3]*rwidth,center[3,0]-scale[3]*rwidth,
            #    2*rwidth*scale[3],2*rwidth*scale[3])
            
            self.circ_tl.setRect(-center[3,1]-scale[3]*rwidth,-center[3,0]-scale[3]*rwidth,
                2*rwidth*scale[3],2*rwidth*scale[3])
            self.circ_tr.setRect(-center[2,1]-scale[2]*rwidth,-center[2,0]-scale[2]*rwidth,
                2*rwidth*scale[2],2*rwidth*scale[2])
            self.circ_bl.setRect(-center[1,1]-scale[1]*rwidth,-center[1,0]-scale[1]*rwidth,
                2*rwidth*scale[1],2*rwidth*scale[1])
            self.circ_br.setRect(-center[0,1]-scale[0]*rwidth,-center[0,0]-scale[0]*rwidth,
                2*rwidth*scale[0],2*rwidth*scale[0])


            full_center = np.mean(center,axis=0)

            self.crossx.setLine(full_center[1]-width/80,full_center[0],
                full_center[1]+width/80,full_center[0])
            self.crossy.setLine(full_center[1],full_center[0]-width/80,
                full_center[1],full_center[0]+width/80)


        self.pix_size_text.setText('Pixel size: %.2f microns' 
                % pixSize)




    def setup_viewbox(self, width):
        """
        Helper function to set up viewbox with title
        :param width: image width in pixels (int)
        """
        # lock aspect ratio
        self.view.setAspectLocked(True)
        # update viewbox range
        self.view.setRange(QtCore.QRectF(-width/2., -width/2., width, width))
        # draw a white rectangle that is the same size as the image to show the image boundary
        rect1 = QtGui.QGraphicsRectItem(-width/2., -width/2., width, width)
        rect1.setPen(QtGui.QPen(QtCore.Qt.white, width/50., QtCore.Qt.SolidLine))
        # add the rectangle to the viewbox
        self.view.addItem(rect1)
        # return the rectangle
        return rect1
        
    def update_viewbox(self, width, height):
        """
        Helper function to adjust viewbox settings
        :param width: new width in pixels (int)
        :param height: new height in pixels (int)
        :return:
        """
        # lock aspect ratio
        self.view.setAspectLocked(True)
        
        # set range to new size
        self.view.setRange(QtCore.QRectF(-width/2, -height/2, width, height))
        # update the bounding rectangle
        self.rect.setPen(QtGui.QPen(QtCore.Qt.white, width/256., QtCore.Qt.SolidLine))
        self.rect.setRect(-width/2, -height/2, width, height)
        
        self.rect_bl.setRect(-width/2,-width/2,width/12, width/12)
        self.rect_bl.setPen(QPen(Qt.cyan, width/256, Qt.SolidLine))
        self.rect_br.setRect(width/2-width/12,-width/2,width/12,width/12)
        self.rect_br.setPen(QPen(Qt.darkMagenta, width/256, Qt.SolidLine))
        self.rect_tl.setRect(-width/2,width/2-width/12,width/12,width/12)
        self.rect_tl.setPen(QPen(Qt.red, width/256, Qt.SolidLine))
        self.rect_tr.setRect(width/2-width/12,width/2-width/12,width/12,width/12)
        self.rect_tr.setPen(QPen(Qt.green, width/256, Qt.SolidLine))


        
        self.crossx0.setLine(-width/80,0,width/80,0)
        self.crossy0.setLine(0,-width/80,0,width/80)
        self.crossx0.setPen(QPen(Qt.green, width/256, Qt.SolidLine))
        self.crossy0.setPen(QPen(Qt.green, width/256, Qt.SolidLine))

        
        self.crossx.setLine(-width/80,0,width/80,0)
        self.crossy.setLine(0,-width/80,0,width/80)
        self.crossx.setPen(QPen(Qt.red, width/256, Qt.SolidLine))
        self.crossy.setPen(QPen(Qt.red, width/256, Qt.SolidLine))

        
        self.circ_tl.setRect(-width/2,width/2-width/40,width/40,width/40)
        self.circ_tl.setPen(QPen(Qt.red, width/256, Qt.SolidLine))
        self.circ_tr.setRect(width/2-width/40,width/2-width/40,width/40,width/40)
        self.circ_tr.setPen(QPen(Qt.green, width/256, Qt.SolidLine))
        self.circ_bl.setRect(-width/2,-width/2,width/40,width/40)
        self.circ_bl.setPen(QPen(Qt.cyan, width/256, Qt.SolidLine))
        self.circ_br.setRect(width/2-width/40,-width/2,width/40,width/40)
        self.circ_br.setPen(QPen(Qt.darkMagenta, width/256, Qt.SolidLine))
        self.pix_size_text.setPos(-width/2+width/7,-width/2+width/2048)

class StripChart:
    """
    Class for displaying time series data
    """

    def __init__(self, canvas, ylabel):
        """
        Initialize a StripChart
        :param canvas: pg.GraphicsLayoutWidget
            Canvas where the plot will be added. Should not contain any other widgets.
        :param ylabel: str
            label for y-axis
        """
        # font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')

        # set canvas as attribute
        self.canvas = canvas
        # generate plot
        self.plotWidget = self.canvas.addPlot()

        # label plot axes
        PlotUtil.label_plot(self.plotWidget, 'Time (s)', ylabel)

        # add grid
        self.plotWidget.showGrid(x=True,y=True,alpha=.8)
        # initialize dictionary of lines to plot
        self.lines = {}

        # set color order
        self.color_order = ['r', 'c', 'm', 'g', 'b', 'y']

        # default time range (seconds)
        self.time_range = 10

    def addSeries(self, series_keys, series_labels):
        """
        Method to define plot series. May or may not be ok to call this method more than once...
        :param series_keys: list of strings
            keys for lines dictionary
        :param series_labels: list of strings
            legend labels for lines
        :return:
        """

        
        if len(series_keys) > 1:
            # add a LegendItem
            legend = self.plotWidget.addLegend()
        else:
            legend = None

        # loop through keys and make PlotItems
        for num, key in enumerate(series_keys):
            # make a PlotItem
            self.lines[key] = self.plotWidget.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(self.color_order[num], width=5),name=series_labels[num])

        if legend is not None:
            # set up the legend
            PlotUtil.setup_legend(legend)

    def set_time_range(self, time_range):
        """
        Method to change the time range for the plot. Could connect this to a callback.
        :param time_range: float
            Time range for the plot in seconds
        :return:
        """
        # set time range attribute
        self.time_range = time_range

    def update_plots(self, time_stamps, **data):
        """
        Method to update the plots with incoming data
        :param time_stamps: (N) ndarray
            image PV timestamps
        :param data: (N) ndarrays
            1D arrays with keywords corresponding to those in lines dict
        :return:
        """

        # filter out any data that doesn't exist yet
        mask = np.logical_not(np.isnan(time_stamps))

        # get current time
        now = datetime.now()
        now_stamp = datetime.timestamp(now)
        # subtract current time from timestamps
        time_stamps = time_stamps - now_stamp
        # mask out invalid data
        time_stamps = time_stamps[mask]

        # loop through datasets
        for key, value in data.items():

            # filter data with mask
            filtered_value = value[mask]

            # find any nan's in the values
            plot_mask = np.logical_not(np.isnan(filtered_value))

            try:
                # set plot data
                self.lines[key].setData(time_stamps[plot_mask], filtered_value[plot_mask])
            except KeyError:
                # catch exceptions related to the wrong key
                print('Data had the wrong name')

        # reset plot range
        self.plotWidget.setXRange(-self.time_range, 0)


class NewPlot(QPlot, Ui_Plot):
    """
    Class for making AMI-style plots in a separate window
    """

    def __init__(self, parent, key_list):
        """
        Create NewPlot object
        :param parent: parent is the App object
        """
        super(NewPlot, self).__init__(parent)
        self.setupUi(self)

        # add parent attribute
        self.parent = parent

        # make a new pyqtgraph plot
        self.plot_item = self.canvas.addPlot(row=0, col=0)
        # self.legend = self.plot_item.addLegend()
        self.plot_item.showGrid(x=True, y=True, alpha=.7)
        # self.plot_ref = self.plot_item.plot(np.array([0]),np.array([0]),
        #        pen=pg.mkPen('b',width=5),symbol='o',symbolBrush='b',
        #        name='reference')
        self.plot_data = self.plot_item.plot(np.linspace(0, 99, 100), np.zeros(100),
                                             pen=pg.mkPen('r', width=5), symbol='o', symbolBrush='r',
                                             name='current data')

        PlotUtil.label_plot(self.plot_item, '', '')

        # placeholder for reference plot
        # self.plot_ref = None

        # get information from the plot window
        self.minimum = float(self.min_lineEdit.text())
        self.maximum = float(self.max_lineEdit.text())
        self.points = float(self.points_lineEdit.text())
        self.update_bins()

        # connect plot window buttons
        self.min_lineEdit.returnPressed.connect(self.update_min)
        self.max_lineEdit.returnPressed.connect(self.update_max)
        self.points_lineEdit.returnPressed.connect(self.update_points)
        self.xaxis_comboBox.activated.connect(self.update_axes)
        self.yaxis_comboBox.activated.connect(self.update_axes)
        self.actionChange_title.triggered.connect(self.change_title)
        # self.actionReference.triggered.connect(self.add_reference)

        # flag to keep track of if a selection has been made yet in a combo box
        self.flag = 1

        # populate combo boxes
        self.populate_combobox(key_list)

        # initialize plot data
        self.xplotdata = None
        self.yplotdata = None
        ## font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')

        self.update_axes()

    def populate_combobox(self, axis_list):
        """
        Method to populate x and y data combo boxes
        :param axis_dict: dictionary of data
        """

        # check if a selection has been made yet. If not, do nothing
        if self.flag:
            # get all the keys from the dictionary and populate the combo boxes
            # axis_list = []
            # for key in axis_dict.keys():
            #    axis_list.append(key)
            self.xaxis_comboBox.clear()
            self.xaxis_comboBox.addItems(sorted(axis_list))
            self.yaxis_comboBox.clear()
            self.yaxis_comboBox.addItems(sorted(axis_list))

    def change_title(self):
        """
        Method to change the plot title. Called from the file menu.
        :return:
        """

        # get input from a dialog
        text, ok = QtGui.QInputDialog.getText(self, 'Change Title', 'Enter title:')

        # set the title if something was entered
        if ok:
            self.setWindowTitle(text)

    def update_min(self):
        """
        Update the x-axis minimum. Called when enter is pressed.
        """
        self.minimum = float(self.min_lineEdit.text())
        self.update_bins()

    def update_max(self):
        """
        Update the x-axis maximum. Called when enter is pressed.
        """
        self.maximum = float(self.max_lineEdit.text())
        self.update_bins()

    def update_points(self):
        """
        Update the number of bins. Called when enter is pressed.
        """
        self.points = float(self.points_lineEdit.text())
        self.update_bins()

    def update_bins(self):
        # calculate bin parameters
        bin_width = (self.maximum - self.minimum) / (self.points - 1)
        dx = bin_width / 2.
        self.bins = np.linspace(self.minimum - dx, self.maximum + dx, int(self.points) + 1)
        self.binPlot = (self.bins[:-1] + self.bins[1:]) / 2.

    def update_axes(self):
        """
        Called when a selection is made in the combo boxes. Updates plot labels and sets keys for data access.
        """
        self.xaxis = str(self.xaxis_comboBox.currentText())
        self.yaxis = str(self.yaxis_comboBox.currentText())
        self.label_plot(self.plot_item, self.xaxis, self.yaxis)
        # disable re-population of combo boxes by setting the flag to 0.
        self.flag = 0

    def label_plot(self, plot, xlabel, ylabel):
        """
        Helper function to set plot labels
        :param plot: pyqtgraph plot item
        :param xlabel: x-axis label (str)
        :param ylabel: y-axis label (str)
        """
        xaxis = plot.getAxis('bottom')
        xaxis.setLabel(text=xlabel,**self.labelStyle)
        xaxis.tickFont = self.font
        xaxis.setPen(pg.mkPen('w',width=1))
        yaxis = plot.getAxis('left')
        yaxis.setLabel(text=ylabel,**self.labelStyle)
        yaxis.tickFont = self.font
        yaxis.setPen(pg.mkPen('w',width=1))

    def update_plot(self, data_dict, key_list):
        """
        Called from the main GUI when new data comes in.
        :param data_dict: dictionary containing the data to plot
        """

        try:
            # if we're starting a new data analysis process, see if we should update combo box
            if data_dict['counter'] == 0:
                self.populate_combobox(key_list)
            # get data based on x and y-axis keys

            if self.xaxis == 'timestamps':

                # get current time
                now = datetime.now()
                now_stamp = datetime.timestamp(now)
                # subtract current time from timestamps
                time_stamps = data_dict['timestamps'] - now_stamp
                # mask out invalid data
                xdata = time_stamps
            else:
                xdata = data_dict[self.xaxis]

            ydata = data_dict[self.yaxis]

            # bin_width = (self.maximum - self.minimum)/(self.points-1)
            # dx = bin_width/2.
            # bins = np.linspace(self.minimum-dx,self.maximum+dx,self.points+1)
            # binPlot = (bins[:-1]+bins[1:])/2.

            # mask out nan's

            mask = np.logical_not(np.isnan(xdata))
            mask = np.logical_and(mask, np.logical_not(np.isnan(ydata)))
            xdata = xdata[mask]
            ydata = ydata[mask]

            # figure out which bin each xdata belongs to
            digitized = np.digitize(xdata, self.bins)

            # ignore any warnings about nan's
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # calculate y-values for each bin
                bin_means = [ydata[digitized == i].mean() for i in range(1, len(self.bins))]

            # create a mask to avoid nan's
            mask = np.logical_not(np.isnan(bin_means))

            self.xplotdata = np.array(self.binPlot)[mask]
            self.yplotdata = np.array(bin_means)[mask]
            # update the plot
            self.plot_data.setData(self.xplotdata, self.yplotdata)

        except:
            # print that there was an error if something didn't work.
            print('error')

    def closeEvent(self, event):
        """
        Called when the window is closed
        :param event: close event
        """

        # remove plot from App's plot list
        self.parent.plots.remove(self)


class Config(QConfig, Ui_Config):
    """
    Class for creating config files
    """

    def __init__(self, parent, filename=None):
        """
        Create Config object
        :param parent: parent is the App object
        """
        super(Config, self).__init__(parent)
        self.setupUi(self)

        self.parent = parent

        self.buttonBox.accepted.connect(self.save_config)

        self.filename = filename
        if filename is not None:
            self.load_config()

    def load_config(self):
        """
        Method that populates dialog with entries from a previously
        created config file.
        """
        # get information from file
        pars = wfs_utils.parse_wfs_config_gui(self.filename)

        # get only the name of the file
        name = self.filename.split('/')[-1][:-4]
        self.lineEdit_filename.setText(name)

        # set lineEdits based on what's in file
        self.lineEdit_energy.setText(str(pars['energy']))
        self.lineEdit_ROI.setText(', '.join(map(str, pars['roi'])))
        self.lineEdit_lineout.setText(str(int(pars['lineout_width'])))
        self.lineEdit_fraction.setText(str(pars['fraction']))
        self.lineEdit_threshold.setText(str(pars['thresh']))
        self.lineEdit_downsampling.setText(str(pars['downsample']))
        self.lineEdit_order.setText(str(pars['order']))
        self.lineEdit_z0.setText(str(pars['z0']))
        self.lineEdit_zf.setText(str(pars['zf']))
        pix = str(pars['pixel'] * 1e6)
        self.lineEdit_pix.setText(pix)
        self.lineEdit_grating_motor.setText(pars['grating_z'])
        self.lineEdit_det_motor.setText(pars['det_z'])
        pitch = str(pars['pitch'] * 1e6)
        self.lineEdit_pitch.setText(pitch)
        self.lineEdit_detName.setText(pars['detName'])
        self.lineEdit_rotation.setText(str(pars['angle']))
        self.lineEdit_update.setText(str(pars['update_events']))

        epics_list = pars['epics_keys']
        epics_text = '\n'.join(epics_list)
        self.epics_TextEdit.setPlainText(epics_text)

    def save_config(self):
        """
        Method that runs when the "Ok" button is clicked. Saves config file
        """
        # make a config parser
        config_parser = ConfigParser.ConfigParser()

        # set all the parameters based on what has been entered
        config_parser.add_section('Main')
        config_parser.set('Main', 'hutch', self.parent.hutch)
        config_parser.set('Main', 'exp_name', self.parent.experiment)
        config_parser.set('Main', 'energy', str(self.lineEdit_energy.text()))
        config_parser.set('Main', 'live', str(self.parent.liveCheckBox.isChecked()))
        config_parser.add_section('Processing')
        roi = [x.strip() for x in str(self.lineEdit_ROI.text()).split(',')]
        config_parser.set('Processing', 'xmin', roi[0])
        config_parser.set('Processing', 'xmax', roi[1])
        config_parser.set('Processing', 'ymin', roi[2])
        config_parser.set('Processing', 'ymax', roi[3])
        config_parser.set('Processing', 'pad', '1')
        config_parser.set('Processing', 'lineout_width',
                          str(self.lineEdit_lineout.text()))
        config_parser.set('Processing', 'fraction',
                          str(self.lineEdit_fraction.text()))
        config_parser.set('Processing', 'padding', '0')
        config_parser.set('Processing', 'thresh',
                          str(self.lineEdit_threshold.text()))
        config_parser.set('Processing', 'downsample',
                          str(self.lineEdit_downsampling.text()))
        config_parser.set('Processing', 'order',
                          str(self.lineEdit_order.text()))
        config_parser.add_section('Setup')
        config_parser.set('Setup', 'z0', str(self.lineEdit_z0.text()))
        config_parser.set('Setup', 'zf', str(self.lineEdit_zf.text()))
        config_parser.set('Setup', 'pixel',
                          str(self.lineEdit_pix.text()) + 'e-6')
        config_parser.set('Setup', 'grating_z',
                          str(self.lineEdit_grating_motor.text()))
        config_parser.set('Setup', 'det_z',
                          str(self.lineEdit_det_motor.text()))
        config_parser.set('Setup', 'pitch',
                          str(self.lineEdit_pitch.text()) + 'e-6')
        config_parser.set('Setup', 'detName',
                          str(self.lineEdit_detName.text()))
        config_parser.set('Setup', 'angle',
                          str(self.lineEdit_rotation.text()))
        config_parser.add_section('Update')
        config_parser.set('Update', 'update_events',
                          str(self.lineEdit_update.text()))

        # epics keys need to be parsed from textbox, and put into a
        # comma separated string
        epics_text = str(self.epics_TextEdit.toPlainText())
        # convert text to list
        epics_list = [x.strip() for x in epics_text.split('\n')]
        # remove any empty strings
        epics_list = list(filter(None, epics_list))
        # convert to single string with comma separation
        epics_string = ','.join(epics_list)
        config_parser.set('Processing', 'epics_keys', epics_string)

        # get the filename that was entered
        filename = 'config/' + str(self.lineEdit_filename.text()) + '.cfg'

        # write the new config file
        with open(filename, 'wb') as configfile:
            config_parser.write(configfile)

        # make sure this is the config file that is being used now
        self.parent.config = filename


class CrosshairWidget(QCrosshair, Ui_Crosshair):
    """
    Class to define a crosshair widget. The widget consists of two crosshair buttons and 4 LineEdits corresponding
    to crosshair positions.

    Attributes
    ----------
    redButton: QPushButton
        Button for selecting red crosshair
    blueButton: QPushButton
        Button for selecting blue crosshair
    red_x: QLineEdit
        x position of red crosshair
    red_y: QLineEdit
        y position of red crosshair
    blue_x: QLineEdit
        x position of blue crosshair
    blue_y: QLineEdit
        y position of blue crosshair
    lineout_image: LineoutImage
        image widget to connect to
    red_crosshair: Crosshair
        red crosshair object that is displayed on lineout_image
    blue_crosshair: Crosshair
        blue crosshair object that is displayed on lineout_image
    current_crosshair: Crosshair
        Variable to keep track of the active crosshair. Can have values of None, red_crosshair, or blue_crosshair.
    """

    def __init__(self, parent=None):
        """
        Initialize the widget
        :param parent:
        """
        super(CrosshairWidget, self).__init__()
        self.setupUi(self)

        # initialize attributes
        self.red_crosshair = None
        self.blue_crosshair = None
        self.current_crosshair = None

    def connect_image(self, image_widget):
        """
        Method to connect the crosshair widget to an image
        :param image_widget: LineoutImage
            The image to connect to
        :return:
        """
        # create Crosshair objects
        self.red_crosshair = Crosshair('red', self.red_x, self.red_y, image_widget)
        self.blue_crosshair = Crosshair('blue', self.blue_x, self.blue_y, image_widget)

        # connect callbacks
        # crosshair selection
        self.redButton.toggled.connect(self.red_crosshair_toggled)
        self.blueButton.toggled.connect(self.blue_crosshair_toggled)
        # red crosshair position
        self.red_x.returnPressed.connect(self.update_red_crosshair)
        self.red_y.returnPressed.connect(self.update_red_crosshair)
        # blue crosshair position
        self.blue_x.returnPressed.connect(self.update_blue_crosshair)
        self.blue_y.returnPressed.connect(self.update_blue_crosshair)

        self.redCircleButton.toggled.connect(self.red_circle_toggled)
        self.blueCircleButton.toggled.connect(self.blue_circle_toggled)
        self.redCircleDiameter.returnPressed.connect(self.set_red_diameter)
        self.blueCircleDiameter.returnPressed.connect(self.set_blue_diameter)

        # diameter in mm
        try:
            self.red_diameter = float(self.redCircleDiameter.text())*1e3
        except ValueError:
            self.red_diameter = 0
        try:
            self.blue_diameter = float(self.blueCircleDiameter.text())*1e3
        except ValueError:
            self.blue_diameter = 0

        #self.red_x.returnPressed.connect(self.update_crosshair(self.red_crosshair))
        #self.red_y.returnPressed.connect(self.update_crosshair(self.red_crosshair))
        #self.blue_x.returnPressed.connect(self.update_crosshair(self.blue_crosshair))
        #self.blue_y.returnPressed.connect(self.update_crosshair(self.blue_crosshair))

    def calculate_distance(self):

        try:
            red_x = float(self.red_x.text())
        except ValueError:
            red_x = 0
        try:
            red_y = float(self.red_y.text())
        except ValueError:
            red_y = 0
        try:
            blue_x = float(self.blue_x.text())
        except ValueError:
            blue_x = 0
        try:
            blue_y = float(self.blue_y.text())
        except ValueError:
            blue_y = 0

        distance = np.sqrt((red_x-blue_x)**2 + (red_y-blue_y)**2)*1e-3

        #self.distanceLineEdit.setText('%.2f' % distance)
        self.distanceLabel.setText('Distance between crosshairs: %.2e mm' % distance)
        #pass

    def update_red_crosshair(self):
        self.calculate_distance()
        self.red_crosshair.update_position()

    def update_blue_crosshair(self):
        self.calculate_distance()
        self.blue_crosshair.update_position()

    def red_circle_toggled(self, evt):
        if evt:
            self.red_crosshair.addCircle(self.red_diameter)
        else:
            self.red_crosshair.removeCircle()

    def blue_circle_toggled(self, evt):
        if evt:
            self.blue_crosshair.addCircle(self.blue_diameter)
        else:
            self.blue_crosshair.removeCircle()

    def set_red_diameter(self):
        try:
            self.red_diameter = float(self.redCircleDiameter.text())*1e3
        except ValueError:
            self.red_diameter = 0
        self.red_crosshair.updateDiameter(self.red_diameter)

    def set_blue_diameter(self):
        try:
            self.blue_diameter = float(self.blueCircleDiameter.text())*1e3
        except ValueError:
            self.blue_diameter = 0
        self.blue_crosshair.updateDiameter(self.blue_diameter)

    def red_crosshair_toggled(self, evt):
        """
        Method that is called when redButton is pressed
        :param evt: bool from signal
            True if redButton is checked, False if not
        :return:
        """

        if evt:
            # if red button is now checked
            # check if blue button is checked
            if self.blueButton.isChecked():
                # if so, uncheck it
                self.blueButton.toggle()
            # update current crosshair
            self.current_crosshair = self.red_crosshair
        else:
            # if red button is now unchecked, set current crosshair to None
            self.current_crosshair = None

    def blue_crosshair_toggled(self, evt):
        """
        Method that is called when blueButton is pressed
        :param evt: bool from signal
            True if blueButton is checked, False if not
        :return:
        """
        if evt:
            # if blue button is now checked
            # check if red button is checked
            if self.redButton.isChecked():
                # if so, uncheck it
                self.redButton.toggle()
            # update current crosshair
            self.current_crosshair = self.blue_crosshair
        else:
            # if red button is now unchecked, set current crosshair to None
            self.current_crosshair = None

    def update_crosshair_coords(self, coords):
        """
        Method to define new crosshair location based on mouseclick.
        :param evt: mouse click event
            Contains scene position
        :return:
        """
        # update current crosshair coordinates based on mouse click location
        if self.current_crosshair is not None:
            # update text to display crosshair location (in whatever units the viewbox coordinates are in)
            self.current_crosshair.xLineEdit.setText('%.1f' % coords.x())
            self.current_crosshair.yLineEdit.setText('%.1f' % coords.y())
            # draw crosshair at new location
            self.current_crosshair.update_position()

            # re-calculate crosshair distance
            self.calculate_distance()

    def update_crosshair_width(self):
        """
        Method to update the width of both crosshairs. Called when image changes shape
        :return:
        """
        # call update width for both crosshairs
        self.red_crosshair.update_width()
        self.blue_crosshair.update_width()


class AverageWidget(QAverageWidget, Ui_AverageWidget):
    """
    Class to define a widget for averaging images.
    """
    def __init__(self, parent=None):
        super(AverageWidget, self).__init__()
        self.setupUi(self)
        
        # connect callbacks
        self.averagingCheckBox.toggled.connect(self.set_averaging)
        self.numImagesLineEdit.returnPressed.connect(self.update_average)

        # set attributes
        self.averaging = self.averagingCheckBox.isChecked()
        # default number of images (for no averaging)
        self.numImages = int(self.numImagesLineEdit.text())
        self.set_averaging()

    def update_average(self):
        if self.averaging:
            self.numImages = int(self.numImagesLineEdit.text())

    def set_averaging(self):
        self.averaging = self.averagingCheckBox.isChecked()
       
        if self.averaging:
            self.update_average()
        else:
            self.numImages = 1

    def get_numImages(self):
        return self.numImages


class LevelsWidget(QLevelsWidget, Ui_LevelsWidget):
    """
    Class to define a widget for adjusting image levels
    """

    def __init__(self, parent=None):
        """
        Initialize the widget.
        :param parent:
        """
        super(LevelsWidget, self).__init__()
        self.setupUi(self)

    def setText(self, minimum, maximum):
        """
        Method to update the text when autoscaling.
        :param minimum: int
            value to set in minLineEdit
        :param maximum: int
            value to set in maxLineEdit
        :return:
        """
        # set the text
        self.minLineEdit.setText('%d' % minimum)
        self.maxLineEdit.setText('%d' % maximum)


class Crosshair:
    """
    Class to represent a crosshair on an image. Draws crosshair on an image widget.
    """

    def __init__(self, color, xLineEdit, yLineEdit, lineout_image):
        """
        Method to create a crosshair
        :param color: str
            Must be a color defined in Qt package
        :param xLineEdit: QLineEdit
            LineEdit corresponding to horizontal position
        :param yLineEdit: QLineEdit
            LineEdit corresponding to vertical position
        :param rect: QGraphicsRectItem
            bounding rectangle of the viewbox
        :param lineout_image: LineoutImage
            widget where the crosshair goes
        """

        # try to set the color based on string input
        try:
            self.color = getattr(Qt, color)
        except AttributeError:
            # default to red
            self.color = Qt.red

        # set some attributes
        self.xLineEdit = xLineEdit
        self.yLineEdit = yLineEdit
        self.lineout_image = lineout_image

        # initialize crosshair coordinates
        self.xPos = 1024
        self.yPos = 1024

        # define lines that define the crosshair
        self.crossh = QtWidgets.QGraphicsLineItem(1024 - 25, 1024, 1024 + 25, 1024)
        self.crossv = QtWidgets.QGraphicsLineItem(1024, 1024 - 25, 1024, 1024 + 25)
        self.crossh.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))
        self.crossv.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))

        self.circle = QtWidgets.QGraphicsEllipseItem(1024, 1024, 0, 0)
        self.circle.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))

        self.diameter = 0

        # put it in the viewbox
        self.lineout_image.view.addItem(self.crossh)
        self.lineout_image.view.addItem(self.crossv)

    def addCircle(self, diameter):
        self.lineout_image.view.addItem(self.circle)

    def removeCircle(self):
        self.lineout_image.view.removeItem(self.circle)

    def updateDiameter(self, diameter):
        self.diameter = diameter
        radius = self.diameter/2
        self.circle.setRect(self.xPos-radius, self.yPos - radius, diameter, diameter)

    def update_width(self):
        """
        Method to update crosshair size if viewbox changed
        :return:
        """
        # get width of the bounding rect
        rect_width = self.lineout_image.rect.boundingRect().width()
        # set line thickness to 1% of the viewbox width
        thickness = rect_width * .01
        # update lines
        self.crossh.setPen(QtGui.QPen(self.color, thickness, Qt.SolidLine))
        self.crossv.setPen(QtGui.QPen(self.color, thickness, Qt.SolidLine))
        self.circle.setPen(QtGui.QPen(self.color, thickness, Qt.SolidLine))

        try:
            # try to get the position of the crosshair
            self.xPos = float(self.xLineEdit.text())
            self.yPos = float(self.yLineEdit.text())
        except ValueError:
            # if it didn't work put it in the corner
            self.xPos = -rect_width/2
            self.yPos = -rect_width/2
        # set width of crosshair to 4% of the viewbox width
        self.crossh.setLine(self.xPos - rect_width * .02, self.yPos,
                             self.xPos + rect_width * .02, self.yPos)
        self.crossv.setLine(self.xPos, self.yPos - rect_width * .02,
                             self.xPos, self.yPos + rect_width * .02)

        radius = self.diameter/2
        self.circle.setRect(self.xPos - radius, self.yPos - radius, self.diameter, self.diameter)

    def update_position(self):
        """
        Method to move the crosshair
        :return:
        """
        rect_width = self.lineout_image.rect.boundingRect().width()
        try:
            self.xPos = float(self.xLineEdit.text())
            self.yPos = float(self.yLineEdit.text())
        except ValueError:
            # if it didn't work put it in the corner
            self.xPos = -rect_width / 2
            self.yPos = -rect_width / 2
        # move crosshair
        self.crossh.setLine(self.xPos - rect_width*.02, self.yPos, self.xPos + rect_width*.02, self.yPos)
        self.crossv.setLine(self.xPos, self.yPos - rect_width*.02, self.xPos, self.yPos + rect_width*.02)
        radius = self.diameter/2
        self.circle.setRect(self.xPos - radius, self.yPos - radius, self.diameter, self.diameter)


