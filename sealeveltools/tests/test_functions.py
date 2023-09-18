from sealeveltools.sl_class import *
from sealeveltools.tests.load_test_files import *


def load_testfiles():
    test_file_dir='/home/oelsmann/Julius/Scripts/sealeveltools/sealeveltools/tests/files/'
    xrat=xr.open_dataset(test_file_dir+'xra.nc')
    xrt=xr.open_dataset(test_file_dir+'xr.nc')
    pdt=pd.read_csv(test_file_dir+'pd')
    flt=1.1
    return [xrat,xrt,pdt,flt]

def test_sl(func):
    """
    tests freshly installed functions
    
    """
    typen=load_testfiles()
    
    for typ in typen:
        #print(typ)
        method_out = getattr(sl(typ), func)
    print('Method '+ func+' works')

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                           
