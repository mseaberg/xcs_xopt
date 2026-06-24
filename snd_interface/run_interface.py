from WFS_interface import App
import warnings
from pyqtgraph.Qt import QtGui
import sys
import argparse

def process_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', action='store', default='cxix28716')

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args

if __name__ == '__main__':

    parsed_args, unparsed_args = process_cl_args()
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    qt_args = sys.argv[:1] + unparsed_args
    app = QtGui.QApplication(qt_args)
    thisapp = App(args=parsed_args)
    thisapp.show()
    sys.exit(app.exec_())
