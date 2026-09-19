import sys


class Utils(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def standard_error(error_data):
        try:
            exc_type, exc_obj, exc_tb = error_data
            return \
                'ERROR: ' + exc_type.__name__ + ': ' + str(exc_obj) + '\nFILE: ' + exc_tb.tb_frame.f_code.co_filename + \
                '\nMETHOD: ' + exc_tb.tb_frame.f_code.co_name + \
                '\nLINE: ' + str(exc_tb.tb_lineno) + \
                '\n------------------------------------------------------------------------'
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return \
                'ERROR: ' + exc_type.__name__ + ': ' + str(exc_obj) + '\nFILE: ' + exc_tb.tb_frame.f_code.co_filename + \
                '\nMETHOD: ' + exc_tb.tb_frame.f_code.co_name + \
                '\nLINE: ' + str(exc_tb.tb_lineno) + \
                '\n------------------------------------------------------------------------'
