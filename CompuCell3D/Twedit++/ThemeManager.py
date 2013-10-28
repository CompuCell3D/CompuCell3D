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
        print '_str=',_str
        r=int(_str[0:2],16)
        g=int(_str[2:4],16)
        b=int(_str[4:6],16)
        print 'r,g,b=',(r,g,b)
        # sys.exit()
        try:
            return QColor(int(_str[0:2],16),int(_str[2:4],16),int(_str[4:6],16))
        except ValueError,e:
            return None
        
        
        
    def  npStrToSciColor(self,_str):   
        print '_str=',_str
        # val = (int(_str[4:6],16)<<16)+(int(_str[2:4],16)<<8)+(int(_str[0:2],16))
        # val = int(_str[0:2],16)<<16+int(_str[2:4],16)<<8+int(_str[4:6],16)
        
        # print 'val=',val
        # sys.exit()
        try:
            return (int(_str[4:6],16)<<16)+(int(_str[2:4],16)<<8)+(int(_str[0:2],16))
        except ValueError,e:
            return None
        
    def applyThemeToEditor(self,_themeName,_editor):
        try:    
            theme=self.themeDict[_themeName]
        except LookupError,e:
            print type(_themeName)
            print 'Could not find theme: '+_themeName+ ' in ThemeManager'
            print 'got these themes=',self.themeDict.keys()
            return
        
            
        lexer=_editor.lexer()        
        
        if not lexer:return
        
        print 'looking for language ',lexer.language()    
        lexerLanguage=str(lexer.language())
        
        print 'lexerName=',lexerLanguage
        print 'theme=',theme.name
        print 'theme.lexerStyleDict.keys()=',theme.lexerStyleDict.keys()
        lexerStyle=theme.getLexerStyle(lexerLanguage.lower())
        
        print 'lexerStyle=',lexerStyle
        print 'theme.lexerStyleDict=',theme.lexerStyleDict
        
        if not lexerStyle: return
        
        #applying global styles
        
        
        
        N2C=self.npStrToQColor
        N2S=self.npStrToSciColor
        
        
        
        
        defaultStyle=theme.getGlobalStyle('Default Style')
        if defaultStyle:
            lexer.setPaper(N2C(defaultStyle.bgColor))
            # _editor.setCaretForegroundColor(N2C(caretStyle.fgColor))            
        
        
        
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
        
        
        #margin color
        lineNumMargStyle=theme.getGlobalStyle('Line number margin')
        if lineNumMargStyle:        
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,lineNumMargStyle.styleID,N2S(lineNumMargStyle.fgColor)) 
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,lineNumMargStyle.styleID,N2S(lineNumMargStyle.bgColor)) 
            # _editor.SendScintilla(QsciScintilla.SCI_SETMARGINTYPEN,1,QsciScintilla.SC_MARGIN_BACK)           
        # _editor.SendScintilla(QsciScintilla.SCI_SETMARGINTYPEN,0,QsciScintilla.SC_MARGIN_FORE)           
        
        
        
        # N2S('FF0000')
        
        for wordStyle in lexerStyle.wordStyles:
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,wordStyle.styleID,N2S(wordStyle.fgColor))    
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,wordStyle.styleID,N2S(wordStyle.bgColor))    
            # # # if wordStyle.styleID==1:
                # # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,wordStyle.styleID,int('FF000',16))       
                # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,wordStyle.styleID,N2S('FF0000'))       
            # # # else:
            
                # # # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,wordStyle.styleID,int(wordStyle.fgColor,16))       
            # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,wordStyle.styleID,int(wordStyle.bgColor,16)) 

            # if wordStyle.styleID==0:
                # # lexer.setDefaultPaper(N2C(wordStyle.bgColor))
                # lexer.setPaper(N2C(wordStyle.bgColor))
                # # _editor.SendScintilla(QsciScintilla.SCI_SETWHITESPACEBACK,True,int(wordStyle.bgColor,16))       
                # # _editor.SendScintilla(QsciScintilla.SCI_SETWHITESPACEFORE,True,int(wordStyle.bgColor,16))     
                # # SCI_SETWHITESPACEBACK(bool useWhitespaceBackColour, int colour)
                # # _editor.setPaper(N2C(wordStyle.bgColor))
            # # elif wordStyle.styleID==1:    
            # else:    
                # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,wordStyle.styleID,int(wordStyle.fgColor,16))       
                # _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK,wordStyle.styleID,int(wordStyle.bgColor,16)) 
                
            
        
        # sys.exit()
        

            
        # _editor.setFont(self.baseFont)              
        # # SCI_STYLESETFORE(int styleNumber, int colour)
        # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,1,255)        
        

