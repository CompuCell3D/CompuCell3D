'''
    this class manages color schemes for twedit. It uses Notepad++ configuration files (xml) to describe how Twedit++ displays different languages  
'''
from DOMUtils import DOMBase

class WordStyle(DOMBase):
    def __init__(self,_name=''):
        DOMBase.__init__(self,_name='WordStyle')
        self.attrNameToTypeDict={'name':(str,''),'styleID':(int,-1),'fgColor':(str,''),'bgColor':(str,''),'fontName':(str,''),'fontStyle':(int,0),'fontSize':(int,-1)}
        
        
class LexerStyle(DOMBase):
    def __init__(self,):
        DOMBase.__init__(self,_name='LexerStyle')    
        self.attrNameToTypeDict={'name':(str,''),'desc':(str,''),'ext':(str,'')}
        
        self.wordStyles=[] # list of wordStyle objects

        
class Theme(object):
    def __init__(self,_name='GENERAL THEME'):
        self.name=_name
        self.themeFileName=''
        self.lexerStyleDict={} # {llexerName:lexerStyle} e.g.  {'python':pythonlexerStyle}
        self.globalStyle={} #{name: style}
            
    def addGlobalStyle(self,_style):
        self.globalStyle[_style.name]=_style

    def getGlobalStyle(self,_name):
        try:
            return self.globalStyle[_name]
        except LookupError,e:
            return None
                
    def addLexerStyle(self,_lexerStyle):
        self.lexerStyleDict[_lexerStyle.name.lower()]=_lexerStyle
    
    def getLexerStyle(self,_languageName):
        try:
            return self.lexerStyleDict[_languageName]
        except LookupError,e:
            return None
        
import Configuration        
import sys,os
from PyQt4.Qsci import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *



class ThemeManager(object):
    def __init__(self):
        self.themeDict={}
        self.tweditRootPath = os.path.dirname(Configuration.__file__)
        self.themeDir=os.path.join(self.tweditRootPath,'themes')
        
        # this dictionary translates scintilla lexer language name to the language names used by Notepad++ theme xml files. Usually no translation is necessary but for example c++ has to be translated to cpp to be able to find proper styling
        self.sciltillaLexerToNppTheme={'c++':'cpp','c#':'cs','d':'cpp','fortran77':'fortran','idl':'python','javascript':'cpp','octave':'matlab','pov':'cpp','properties':'props','spice':'vhdl'}
        
    def getThemeNames(self):
        
        themesSorted=sorted(self.themeDict.keys())        
        return themesSorted
        
    def readThemes(self):
        import glob
        
        themeFileList = glob.glob(self.themeDir+"/*.xml")
        for themeFileName in themeFileList:
            coreThemeName,ext=os.path.splitext(os.path.basename(themeFileName))
            theme=Theme(coreThemeName)
            theme.themeFileName=themeFileName
            
            self.parseTheme(_theme=theme)
            self.themeDict[coreThemeName]=theme
        # print 'themeFileList   =',themeFileList 
        # print 'self.themeDict=',self.themeDict
        # sys.exit()
    def parseTheme(self,_theme):            
        from xml.dom.minidom import parse,parseString
        
        dom=parse(_theme.themeFileName)
        lexerStylesElems=dom.getElementsByTagName('LexerStyles')
        # lexerStylesElems=dom1.getElementsByTagName('LexerStyles')

        lexerStylesElem=lexerStylesElems[0]

        # print lexerStylesElem

        lexerTypeElems=lexerStylesElem.getElementsByTagName('LexerType')
        # print 'lexerTypeElems=',lexerTypeElems

        lexerTypes=[]
        
        for lexerTypeElem in lexerTypeElems:
            lexerStyle=LexerStyle()
            lexerStyle.fromDOMElem(lexerTypeElem)
            
            # lexerTypes.append(LexerType())
            # lexerType=lexerTypes[-1]
            # lexerType.fromDOMElem(lexerTypeElem)
            # print 'lexerType=',lexerType
            
            wordsStyleElems=lexerTypeElem.getElementsByTagName('WordsStyle')
            
            
            for wordsStyleElem in wordsStyleElems:
                lexerStyle.wordStyles.append(WordStyle())
                wordStyle=lexerStyle.wordStyles[-1]
                wordStyle.fromDOMElem(wordsStyleElem)
                
            _theme.addLexerStyle(lexerStyle)
            
        globalStylesElems=dom.getElementsByTagName('GlobalStyles')
        
        globalStylesElem=globalStylesElems[0]
        
        widgetStyleElems=globalStylesElem.getElementsByTagName('WidgetStyle')
        # print 'widgetStyleElems=',widgetStyleElems
        for widgetStyleElem in widgetStyleElems:
            widgetStyle=WordStyle()
            widgetStyle.fromDOMElem(widgetStyleElem)
            
            _theme.addGlobalStyle(widgetStyle)
                    
        
    def npStrToQColor(self,_str):
        # # # print '_str=',_str
        r=int(_str[0:2],16)
        g=int(_str[2:4],16)
        b=int(_str[4:6],16)
        # # # print 'r,g,b=',(r,g,b)
        # sys.exit()
        try:
            return QColor(int(_str[0:2],16),int(_str[2:4],16),int(_str[4:6],16))
        except ValueError,e:
            return None
        
        
        
    def  npStrToSciColor(self,_str):   
        # # # print '_str=',_str
        # val = (int(_str[4:6],16)<<16)+(int(_str[2:4],16)<<8)+(int(_str[0:2],16))
        # val = int(_str[0:2],16)<<16+int(_str[2:4],16)<<8+int(_str[4:6],16)
        
        # print 'val=',val
        # sys.exit()
        try:
            return (int(_str[4:6],16)<<16)+(int(_str[2:4],16)<<8)+(int(_str[0:2],16))
        except ValueError,e:
            return None
            
    def applyGlobalStyleItems(self,theme,_editor):
        N2C=self.npStrToQColor
        N2S=self.npStrToSciColor
        
        
        # defaultStyle=theme.getGlobalStyle('Global override')
        defaultStyle=theme.getGlobalStyle('Default Style')
        if defaultStyle:
            _editor.setPaper(N2C(defaultStyle.bgColor))
            # for editor with lexers we set paper color for lexer as well otherwise page might have gaps in coloring
            lexer=_editor.lexer()                       
            if lexer:
                lexer.setPaper(N2C(defaultStyle.bgColor))
                
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,defaultStyle.styleID,N2S(defaultStyle.fgColor))             
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,defaultStyle.styleID,N2S(defaultStyle.bgColor)) 
            # since some lexers are using styles which are not defined in npp file it is a good idea to assign first all styles to some reasonable default style
            # later, those styles defined by npp styles can be overwritten
            for styleId in xrange(0,255):
                _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,styleId,N2S(defaultStyle.fgColor))             
                _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,styleId,N2S(defaultStyle.bgColor)) 
                
            
        caretStyle=theme.getGlobalStyle('Caret colour')
        
        
        if caretStyle:
            _editor.setCaretForegroundColor(N2C(caretStyle.fgColor))            
            # _editor.setCaretLineBackgroundColor(self.npStrToQColor(caretStyle.bgColor))
            
        currentLineStyle=theme.getGlobalStyle('Current line background colour')    
        if currentLineStyle:
            _editor.setCaretLineBackgroundColor(N2C(currentLineStyle.bgColor))
            
        foldMarginStyle=theme.getGlobalStyle('Fold margin')        
        if foldMarginStyle:
            _editor.setFoldMarginColors(N2C(foldMarginStyle.fgColor),N2C(foldMarginStyle.bgColor))
            
        lineNumberStyle=theme.getGlobalStyle('Line number margin')        
        if lineNumberStyle:
        
            _editor.setFoldMarginColors(N2C(foldMarginStyle.fgColor),N2C(foldMarginStyle.bgColor))
            
        selectionStyle=theme.getGlobalStyle('Selected text colour')
        if selectionStyle:
            _editor.setSelectionBackgroundColor(N2C(selectionStyle.bgColor))
        
        
        for style in theme.globalStyle.values():
            if style.styleID>0:
                _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,style.styleID,N2S(style.fgColor)) 
                _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,style.styleID,N2S(style.bgColor)) 
                _editor.SendScintilla(QsciScintilla.SCI_STYLESETBOLD,style.styleID,style.fontStyle) 

        #fold style 
        foldStyle=theme.getGlobalStyle('Fold')
        if foldStyle:                    
            foldMarkerEnums=[QsciScintilla. SC_MARKNUM_FOLDEREND , QsciScintilla. SC_MARKNUM_FOLDEROPENMID , QsciScintilla.SC_MARKNUM_FOLDERMIDTAIL,\
            QsciScintilla.SC_MARKNUM_FOLDERTAIL , QsciScintilla.SC_MARKNUM_FOLDERSUB , QsciScintilla.SC_MARKNUM_FOLDER ,  QsciScintilla.SC_MARKNUM_FOLDEROPEN]    
            # to be consistent with notepad++ we switch bgColor with fgColor to style fold markers    
            for foldMarker in foldMarkerEnums:                
                _editor.SendScintilla(QsciScintilla.SCI_MARKERSETFORE, foldMarker , N2S(foldStyle.bgColor)) # notice, marker foreground is styled using bgColor
                _editor.SendScintilla(QsciScintilla.SCI_MARKERSETBACK,foldMarker , N2S(foldStyle.fgColor)) # notice, marker background is styled using fgColor
    
            # # # _editor.SendScintilla(QsciScintilla.SCI_MARKERSETFORE,QsciScintilla. SC_MARKNUM_FOLDEROPENMID , N2S(foldStyle.fgColor))
            # # # _editor.SendScintilla(QsciScintilla.SCI_MARKERSETBACK,QsciScintilla. SC_MARKNUM_FOLDEROPENMID , N2S(foldStyle.bgColor))


            
        # # # #margin color
        # # # lineNumMargStyle=theme.getGlobalStyle('Line number margin')
        # # # if lineNumMargStyle:        
            # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,lineNumMargStyle.styleID,N2S(lineNumMargStyle.fgColor)) 
            # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,lineNumMargStyle.styleID,N2S(lineNumMargStyle.bgColor)) 
            # # # # _editor.SendScintilla(QsciScintilla.SCI_SETMARGINTYPEN,1,QsciScintilla.SC_MARGIN_BACK)           
        # # # # _editor.SendScintilla(QsciScintilla.SCI_SETMARGINTYPEN,0,QsciScintilla.SC_MARGIN_FORE)           
        
        # # # #matching brace color
        # # # matchingBraceStyle=theme.getGlobalStyle('Brace highlight style')
        # # # if matchingBraceStyle:        
            # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,matchingBraceStyle.styleID,N2S(matchingBraceStyle.fgColor)) 
            # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,matchingBraceStyle.styleID,N2S(matchingBraceStyle.bgColor)) 
            # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBOLD,matchingBraceStyle.styleID,matchingBraceStyle.fontStyle) 

        # # # #bad brace color
        # # # badBraceStyle=theme.getGlobalStyle('Bad brace colour')
        # # # if badBraceStyle:        
            # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,badBraceStyle.styleID,N2S(badBraceStyle.fgColor)) 
            # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,badBraceStyle.styleID,N2S(badBraceStyle.bgColor)) 
            # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBOLD,badBraceStyle.styleID,badBraceStyle.fontStyle) 


        # # # #fold style 
        # # # foldStyle=theme.getGlobalStyle('Fold')
        # # # if badBraceStyle:        
            # # # # _editor.SendScintilla(QsciScintilla.SCI_SETFOLDMARGINCOLOUR,1, N2S(foldStyle.bgColor)) 
            # # # _editor.SendScintilla(QsciScintilla.SCI_SETFOLDMARGINHICOLOUR,1, N2S(foldStyle.bgColor)) 
            # # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,badBraceStyle.styleID,N2S(badBraceStyle.bgColor)) 
            # # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBOLD,badBraceStyle.styleID,badBraceStyle.fontStyle) 
            
   # # # # SendMessage hSci, %SCI_StyleSetFore, %Style_BraceLight, %Red    'set brace highlighting color
   # # # # SendMessage hSci, %SCI_StyleSetBack, %Style_BraceLight, %Yellow 'set brace highlighting color
   # # # # SendMessage hSci, %SCI_StyleSetFore, %Style_BraceBad, %Green    'set brace bad color
   # # # # SendMessage hSci, %SCI_StyleSetBack, %Style_BraceBad, %Red     'set brace bad color    
   
    def getStyleFromTheme(self,_styleName,_themeName):    
        N2C=self.npStrToQColor
        N2S=self.npStrToSciColor    
        try:    
            theme=self.themeDict[_themeName]
        except LookupError,e:
            print type(_themeName)
            print 'Could not find theme: '+_themeName+ ' in ThemeManager'
            print 'got these themes=',self.themeDict.keys()
            return None
        
        style=theme.getGlobalStyle(_styleName)
        if style:
            return style
        else:
            return None
   
    
    
    def applyThemeToEditor(self,_themeName,_editor):
        
        N2C=self.npStrToQColor
        N2S=self.npStrToSciColor
    
        try:    
            theme=self.themeDict[_themeName]
        except LookupError,e:
            print type(_themeName)
            print 'Could not find theme: '+_themeName+ ' in ThemeManager'
            print 'got these themes=',self.themeDict.keys()
            return
        
            
        lexer=_editor.lexer()        
        
        if not lexer:
#             print 'APPLYING GLOBAL STYLE ITEMS'
            self.applyGlobalStyleItems(theme,_editor)
            return

        
        # # # print 'looking for language ',lexer.language()    
        lexerLanguage=str(lexer.language())
        
        # # # print 'lexerName=',lexerLanguage
        # # # print 'theme=',theme.name
        # # # print 'theme.lexerStyleDict.keys()=',theme.lexerStyleDict.keys()
        # print 'lexer language=',lexerLanguage.lower()
        try:
            nppStylerLanguageName=self.sciltillaLexerToNppTheme[lexerLanguage.lower()]
        except LookupError,e:
            nppStylerLanguageName=lexerLanguage.lower()
            
        
        lexerStyle=theme.getLexerStyle(nppStylerLanguageName)
        # # # lexerStyle=theme.getLexerStyle(lexerLanguage.lower())
        
        # print 'lexerStyle=',lexerStyle
        # print 'theme.lexerStyleDict=',theme.lexerStyleDict
        
        if not lexerStyle:return
        
        #applying global styles
        
        self.applyGlobalStyleItems(theme,_editor)
        
        
        for wordStyle in lexerStyle.wordStyles:
            # print 'wordStyle.styleID=',wordStyle.styleID
            # print 'wordStyle.fgColor=',wordStyle.fgColor
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,wordStyle.styleID,N2S(wordStyle.fgColor))    
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,wordStyle.styleID,N2S(wordStyle.bgColor))    

        

