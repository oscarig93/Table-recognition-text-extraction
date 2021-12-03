"""
Created on Sat Feb  6 19:58:11 2021

@author: oscarrodriguez
"""
#%%# Import
import cv2
import numpy as np
import os
import TableRecognition
import pandas as pd 
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category= Warning)
import pytesseract
from pdf2image import convert_from_path
from tabulate import tabulate
try: 
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'   
except: pass
#%%# Get pdfs into jpeg and Run OTR 

def pdf2Image(path, newdir):
    '''
    Gets pdfs into jpeg for analysis
    '''
    fecha = path.split('Asis-vot-OFICIAL-')[1][:-4]
    print(fecha)
    try:
        #Create directory to save images
        new_dir = f'{newdir}_{fecha}'
        os.mkdir(new_dir)
        #Covert PDF to image
        try:
            pages = convert_from_path(path, 500, poppler_path= '/usr/local/Cellar/poppler/21.01.0/bin')
        except:
            pages = convert_from_path(path, 500, poppler_path= '/usr/local/Cellar/poppler/20.12.1/bin')
        #Save pages in jpeg format
        for i, page in enumerate(pages):
            if i > 0: # Exclude asistencia
                page.save(f'{new_dir}/ley_{i}.jpg'  , 'JPEG')
    except FileExistsError:
        pass
    return fecha
    
def OTR(filename):
    '''
    Recognizes cells inside table in scanned document. 
    Uses TableRecognition module by ulikohler https://github.com/ulikoehler/OTR
    '''
    img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File {0} does not exist".format(filename))
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgGrey, 150, 255, cv2.THRESH_BINARY_INV)[1]
    imgThreshInv = cv2.threshold(imgGrey, 150, 255, cv2.THRESH_BINARY)[1]    
    imgDil = cv2.dilate(imgThresh, np.ones((5, 5), np.uint8), iterations = 1)
    contour_analyzer = TableRecognition.ContourAnalyzer(imgDil)
    # 1st pass (black in algorithm diagram)
    contour_analyzer.filter_contours(min_area=400)
    contour_analyzer.build_graph()
    contour_analyzer.remove_non_table_nodes()
    contour_analyzer.compute_contour_bounding_boxes()
    contour_analyzer.separate_supernode()
    contour_analyzer.find_empty_cells(imgThreshInv)
    contour_analyzer.find_corner_clusters()
    contour_analyzer.compute_cell_hulls()
    contour_analyzer.find_fine_table_corners()
    # Add missing contours to contour list
    missing_contours = contour_analyzer.compute_filtered_missing_cell_contours()
    contour_analyzer.contours += missing_contours
    # 2nd pass (red in algorithm diagram)
    contour_analyzer.compute_contour_bounding_boxes()
    contour_analyzer.find_empty_cells(imgThreshInv)
    contour_analyzer.find_corner_clusters()
    contour_analyzer.compute_cell_hulls()
    contour_analyzer.find_fine_table_corners()
    # End of 2nd pass. Continue regularly
    contour_analyzer.compute_table_coordinates(xthresh=8., ythresh=20.)
    contour_analyzer.draw_table_coord_cell_hulls(img, xscale=.8, yscale=.8)
    try:
        os.mkdir('fingerprints')
    except FileExistsError:
        pass

    #### FECHA NO FUNCIONA _ CAMBIAR ESTO
    fecha = filename.split('/')[-2]
    num_ley = filename.split('/')[-1][:-4]
    cv2.imwrite('fingerprints/'+ fecha + '_' + num_ley + '.png', img)
    return imgThreshInv, contour_analyzer

#%%# Object

''' I rely on the fact that TableRecogition doesn't confuse x coordinates
(as they are more spaced out in the actual document) to correct the y coordinates
and order document into two congresspeople per row. 
'''

class W():
    def __init__(self, filename):
        imgThreshInv, contour_analyzer = OTR(filename)
        self.img = imgThreshInv
        self.contour_analyzer = contour_analyzer

    def extractCell(self, coords):
        '''
        Extracts section of img given tuple of TableRecognition table coordinates
        '''
        return self.contour_analyzer.extract_cell_from_image(self.img, coords, xscale=1, yscale=1)

    def crop(self, img, by): 
        '''
        Returns cropped image
        '''
        row, col = img.shape
        return img[round(row*by):round(row*(1-by)), round(col*by):round(col*(1-by))]
    
    def cellIsEmpty(self, img_sect):
        '''
        Checks whether the cell is empty. Returns: Boolean
        '''
        # Check proportion of white pixels in each cell
        return True if np.mean(self.crop(img_sect, .25)) > 254 else False
    
    def OCR(self, img_sect, by = 0):
        '''
        Uses tesseract to read strings from a binarized image. Returns: string
        '''
        crop_img = self.crop(img_sect, by)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        border = cv2.copyMakeBorder(crop_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value= [255, 255])
        resizing = cv2.resize(border, None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
        dilation = cv2.dilate(resizing, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        return pytesseract.image_to_string(erosion, lang = 'spa')
    
    def createDf(self):
        '''
        Creates DataFrame with key parameters. Assumes title cell's center divides document'
        '''
        # get table coordinates and cell centers
        tab_coords = self.contour_analyzer.cell_table_coord
        cell_centers =  self.contour_analyzer.cell_centers
        # make data frame 
        df = pd.DataFrame(tab_coords, columns=['x', 'y'])
        df['xy'] = list(zip(df.x, df.y)) # this is used to feed TableRecognition object

        # TODO: use this to break ties later
        df['y_centers'] = cell_centers[:,1]

        # check 1st cell (title) - its x coord divides congress document left and right
        one = tuple(tab_coords[np.where(tab_coords[:,1] == 0)].flatten())
        middle = one[0]
        # middle = int(df.x.iloc[np.where(df.y == 0)])
        df['side'] = np.where(df.x < middle, 'izq', 'der')
        df['congresista'] = list(zip(df.side, df.y))
        asunto = self.OCR(self.extractCell(one))

        # get each cell in table and check whether its empty
        df['img_sect'] = df.apply(lambda row: w.extractCell((row.xy)), axis=1)
        df['empty'] = df.apply(lambda row: w.cellIsEmpty((row.img_sect)), axis=1)
        df.sort_values(by = ['x', 'y'], inplace = True)
        return df, asunto
    
    def fixDf(self, df):    
        '''
        Makes sure that each x-y pair corresponds to a congressperson (4 or 6 cells)
        '''
        # Each unique val should correspond to a congresista
        unique_con = sorted((pd.unique(df.congresista)), reverse = True)
        # Check how long many cols in each row belonging to a congresista and their y center avg
        t = pd.pivot_table(df, values = 'y_centers', index = ['congresista'], aggfunc= ['count', np.mean])
        t.columns = ['num_columnas', 'y_center_mean']
        print(t)
        # if any 'congresista' has 1, 2, 3 or 5 cols -> fake congresista
        if t['num_columnas'].isin([1, 2, 3, 5]).any():

            # Two adj 'congresistas' whose cols add to 4 or 6 make a real congresista
            t['sum'] = t['num_columnas'] + t['num_columnas'].shift(1)            
            change_cong = t[t['sum'].isin([4, 6])].index    
            fix = {cong: unique_con[unique_con.index(cong)+ 1] for cong in change_cong}
            # make fake congresistas real congresistas
            fix = sorted(fix.items(), key = lambda x: (x[0][0], x[0][1]), reverse = True)    
            past_nfake = None
            for fake, nfake in fix:
                if fake == past_nfake: #If two adjacent cells add to 4 or 6 only join the pair further below
                    continue
                df.loc[(df['side'] == fake[0]) & (df['y'] == fake[1]), ['y']] = nfake[1]
                past_nfake = nfake
            df['congresista'] = list(zip(df.side, df.y))

            # for fake, nfake in fix.items():
            #     df.loc[(df['side'] == fake[0]) & (df['y'] == fake[1]), ['y']] = nfake[1]
            # df['congresista'] = list(zip(df.side, df.y))
            # idk why this doesnt work: df['congresista'].replace(fix, inplace = True) - maybe cause congresista is a tuple

        # Only keep real congresistas -> last 65 rows on each side
        g = sorted(pd.unique(df.congresista), key = lambda x: (-x[1]))
        der = [(side, y) for (side, y) in g if side == 'der'][65:]
        izq = [(side, y) for (side, y) in g if side == 'izq'][65:]
        df.drop(df[df['congresista'].isin(der + izq)].index, inplace = True)
        return df

    def extraerVotos(self, df):
        '''
        If congresista voted:
        If rightmost cell not empty -> abstencion, second righmost -> No, third rightmost -> Si
        '''
        # get all congresistas ids ordered
        ids = sorted(pd.unique(df.congresista), key = lambda x: (x[0], -x[1]), reverse = True)
        votos = dict()
        num = 130 # TODO: make numbering easier with enumerate - start from top left
        for cong in ids:    
            # each row corresponds to a cell from a congresista
            df_row = df.iloc[np.where((df.congresista == cong))]
            cols = len(df_row.index)
            # might be more efficient if we get img_sect and empty here  for 1st 3 rows
            # if theres 4 cells -> was out
            if cols == 4:
                votos[num] = 'LICENCIA/AUSENTE'
                # print('\n', num, cong, '\n')
            elif cols == 6:
                empties = tuple(df_row['empty'])
                # print('\n', num, cong, '\n', empties, '\n')
                if sum(empties) > 1: # if 1 or less is empty, somethings wrong
                    if empties[-1] == False: # print('ABSTENCION')
                        votos[num] = 'ABSTENCION'                       
                    elif empties[-2] == False: # print('NO')
                        votos[num] = 'NO'                       
                    elif empties[-3] == False: # print('SI')
                        votos[num] = 'SI'                      
                    else: # print('NO VOTO')
                        votos[num] = 'NO VOTO'                      
                else: #print('np.nan')
                    votos[num] = np.nan
            else:
                votos[num] = np.nan
            num -= 1 
        return votos
    
    def extraerNomPar(self, df):
        '''
        Second column corresponds to congresspersons' partido, third to name
        '''
        ids = sorted(pd.unique(df.congresista), key = lambda x: (x[0], -x[1]), reverse = True)
        congresistas, partidos = dict(), dict()
        congresista = 130
        # second row in df corresponds to partido cell and third to name
        for cong in ids:    
            df_row = df.iloc[np.where((df.congresista == cong))]
            cols = len(df_row.index)
            if cols == 4 or cols == 6:
                partidos[congresista] = w.OCR(df_row.iloc[1].img_sect, by = .15) # cropping a bit helps
                congresistas[congresista] = w.OCR(df_row.iloc[2].img_sect) 
            else:
                partidos[congresista] = np.nan
                congresistas[congresista] = np.nan
            congresista -= 1
        return congresistas, partidos

    def csvOutput(self, congresistas, votos, partidos, asunto, fecha):
        '''
        Returns dataframe with each congressperson's vote for a given law
        '''
        df = pd.DataFrame(list(congresistas.items()), columns = ['index', 'congresistas'])
        df['votos'] = df['index'].map(votos_ley)
        df['partidos'] = df['index'].map(partidos)
        df['asunto'] = np.array([asunto]*len(df.index))
        df['fecha'] =  np.array([fecha]*len(df.index))
        for column in 'congresistas', 'partidos', 'asunto':
            df[column] = df[column].apply(lambda x: x.replace('\n', '') if type(x) == str else x)     
        return df

#%%# Test code 1
"""

 Begin test code 

"""

filename = '/Users/oscarrodriguez/Desktop/CongresoTransparente/OCR/imgs_28-5-20/ley_1.jpg'

w = W(filename)
df, asunto = w.createDf()
dff = w.fixDf(df)
print('where there no change?', df.equals(dff))
votos_ley = w.extraerVotos(dff)
congresistas, partidos = w.extraerNomPar(dff)
result = w.csvOutput(congresistas, votos_ley, partidos, 'prueba', '28-5-20')

print(filename, 'Tabla votos', tabulate(result[['congresistas', 'partidos', 'votos']], 
headers = 'keys', tablefmt = 'orgtbl'), sep = '\n\n', end = '\n\n')
print('Resumen', f'Num congresistas: {len(votos_ley)}', Counter(votos_ley.values()), sep = '\n\n')

# pd.set_option('display.max_rows', None)
# print(pd.pivot_table(df, values = 'y_centers', index = ['congresista'], aggfunc= ['count', np.mean]))

#%%# Test code 2

import time

if __name__ == '__main__':
    start = time.clock()
    pdfsCongreso = [f for f in os.listdir('./pdfs') if f.endswith('.pdf')]
    fechas = []
    for file in pdfsCongreso:
        print(file)
        fecha = pdf2Image('pdfs/' + file, 'imgs')
        fechas.append(fecha)
    print('Finished pdfToImage', '\n\n')
    pagina = 1
    for fecha in fechas:
        os.chdir(f'imgs_{fecha}')
        print(f'fecha: {fecha}', '\n\n')
        for PDFCongreso in [os.path.abspath(f) for f in os.listdir() if f.endswith('.jpg')]:
            w = W(PDFCongreso)
            df, asunto = w.createDf()
            df = w.fixDf(df)
            votos_ley = w.extraerVotos(df)
            congresistas, partidos = w.extraerNomPar(df) 
            # output table to csv
            # TODO: address cases where len(votos) != len(congresistas)
            result = w.csvOutput(congresistas, votos_ley, partidos, asunto, fecha)
            num_ley = PDFCongreso.split('.')[0].split('_')[-1]
            result.to_csv(f'../votos_csv/{fecha}_ley_{num_ley}.csv', index = False)
            # print results
            print(f'filename: {PDFCongreso}', f'ley: {num_ley}', f'Asunto: {asunto}',
            'Tabla votos', tabulate(result[['congresistas', 'partidos', 'votos']], headers = 'keys', tablefmt = 'orgtbl'),
            'Resumen', f'Num congresistas: {len(votos_ley)}', Counter(votos_ley.values()), 
            'CSV saved', '- . ' * 50, sep = '\n''\n', end = '\n''\n''\n''\n')
            pagina += 1

        print('# - ' * 150)
        os.chdir('..')    # TODO: needs updating
        
    end = time.clock()
    print(f'It took {end - start} minutes to process {pagina - 1} page(s)', '\n', 
    f'-> avg time to process one page: {(end - start)/(pagina - 1)} minutes', '\n''\n') 
