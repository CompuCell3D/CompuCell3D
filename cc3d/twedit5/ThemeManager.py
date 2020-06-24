"""
this class manages color schemes for twedit. It uses Notepad++ configuration files (xml)
to describe how Twedit++ displays different languages
"""
from .DOMUtils import DOMBase
from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5 import twedit
from xml.dom.minidom import parse, parseString
import glob


class WordStyle(DOMBase):

    def __init__(self, _name=''):
        DOMBase.__init__(self, _name='WordStyle')

        self.attrNameToTypeDict = {'name': (str, ''), 'styleID': (int, -1), 'fgColor': (str, ''), 'bgColor': (str, ''),
                                   'fontName': (str, ''), 'fontStyle': (int, 0), 'fontSize': (int, -1)}


class LexerStyle(DOMBase):

    def __init__(self, ):
        DOMBase.__init__(self, _name='LexerStyle')

        self.attrNameToTypeDict = {'name': (str, ''), 'desc': (str, ''), 'ext': (str, '')}

        self.wordStyles = []  # list of wordStyle objects


class Theme(object):

    def __init__(self, _name='GENERAL THEME'):

        self.name = _name

        self.themeFileName = ''

        # {llexerName:lexerStyle} e.g.  {'python':pythonlexerStyle}
        self.lexerStyleDict = {}

        # {name: style}
        self.globalStyle = {}

    def addGlobalStyle(self, _style):

        self.globalStyle[_style.name] = _style

    def getGlobalStyle(self, _name):
        try:
            return self.globalStyle[_name]
        except LookupError:
            return None

    def addLexerStyle(self, _lexerStyle):
        self.lexerStyleDict[_lexerStyle.name.lower()] = _lexerStyle

    def getLexerStyle(self, _languageName):

        try:
            return self.lexerStyleDict[_languageName]
        except LookupError:
            return None


class ThemeManager(object):

    def __init__(self):

        self.themeDict = {}

        # self.tweditRootPath = os.path.dirname(Configuration.__file__)
        osp_dir = os.path.dirname
        self.tweditRootPath = osp_dir(osp_dir(twedit.__file__))

        self.themeDir = os.path.join(self.tweditRootPath, 'themes')

        # this dictionary translates scintilla lexer language name to the language names
        # used by Notepad++ theme xml files. Usually no translation is necessary but
        # for example c++ has to be translated to cpp to be able to find proper styling

        self.sciltillaLexerToNppTheme = {'c++': 'cpp', 'c#': 'cs', 'd': 'cpp', 'fortran77': 'fortran', 'idl': 'python',
                                         'javascript': 'cpp', 'octave': 'matlab', 'pov': 'cpp', 'properties': 'props',
                                         'spice': 'vhdl'}

    def getThemeNames(self):

        themesSorted = sorted(self.themeDict.keys())

        return themesSorted

    def readThemes(self):

        theme_file_list = glob.glob(self.themeDir + "/*.xml")

        for themeFileName in theme_file_list:
            core_theme_name, ext = os.path.splitext(os.path.basename(themeFileName))

            theme = Theme(core_theme_name)

            theme.themeFileName = themeFileName

            self.parseTheme(_theme=theme)
            self.themeDict[core_theme_name] = theme

    def parseTheme(self, _theme):

        dom = parse(_theme.themeFileName)

        lexerStylesElems = dom.getElementsByTagName('LexerStyles')

        # lexerStylesElems=dom1.getElementsByTagName('LexerStyles')

        lexerStylesElem = lexerStylesElems[0]

        # print lexerStylesElem

        lexerTypeElems = lexerStylesElem.getElementsByTagName('LexerType')

        # print 'lexerTypeElems=',lexerTypeElems

        lexerTypes = []

        for lexerTypeElem in lexerTypeElems:

            lexer_style = LexerStyle()

            lexer_style.fromDOMElem(lexerTypeElem)

            words_style_elems = lexerTypeElem.getElementsByTagName('WordsStyle')

            for wordsStyleElem in words_style_elems:
                lexer_style.wordStyles.append(WordStyle())
                word_style = lexer_style.wordStyles[-1]
                word_style.fromDOMElem(wordsStyleElem)

            _theme.addLexerStyle(lexer_style)

        global_styles_elems = dom.getElementsByTagName('GlobalStyles')

        global_styles_elem = global_styles_elems[0]

        widget_style_elems = global_styles_elem.getElementsByTagName('WidgetStyle')

        for widgetStyleElem in widget_style_elems:
            widget_style = WordStyle()
            widget_style.fromDOMElem(widgetStyleElem)

            _theme.addGlobalStyle(widget_style)

    def npStrToQColor(self, _str):

        r = int(_str[0:2], 16)

        g = int(_str[2:4], 16)

        b = int(_str[4:6], 16)

        try:
            return QColor(int(_str[0:2], 16), int(_str[2:4], 16), int(_str[4:6], 16))
        except ValueError:
            return None

    def npStrToSciColor(self, _str):

        try:
            return (int(_str[4:6], 16) << 16) + (int(_str[2:4], 16) << 8) + (int(_str[0:2], 16))
        except ValueError:
            return None

    def applyGlobalStyleItems(self, theme, _editor):

        N2C = self.npStrToQColor

        N2S = self.npStrToSciColor

        default_style = theme.getGlobalStyle('Default Style')

        if default_style:

            _editor.setPaper(N2C(default_style.bgColor))

            # for editor with lexers we set paper color for lexer as well otherwise page might have gaps in coloring
            lexer = _editor.lexer()

            if lexer:
                lexer.setPaper(N2C(default_style.bgColor))

            _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE, default_style.styleID, N2S(default_style.fgColor))

            _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK, default_style.styleID, N2S(default_style.bgColor))

            # since some lexers are using styles which are not defined in npp file it is a
            # good idea to assign first all styles to some reasonable default style

            # later, those styles defined by npp styles can be overwritten

            for style_id in range(0, 255):
                _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE, style_id, N2S(default_style.fgColor))

                _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK, style_id, N2S(default_style.bgColor))

        caret_style = theme.getGlobalStyle('Caret colour')

        if caret_style:
            _editor.setCaretForegroundColor(N2C(caret_style.fgColor))

        current_line_style = theme.getGlobalStyle('Current line background colour')

        if current_line_style:
            _editor.setCaretLineBackgroundColor(N2C(current_line_style.bgColor))

        fold_margin_style = theme.getGlobalStyle('Fold margin')
        if fold_margin_style:
            _editor.setFoldMarginColors(N2C(fold_margin_style.fgColor), N2C(fold_margin_style.bgColor))

        line_number_style = theme.getGlobalStyle('Line number margin')
        if line_number_style:
            _editor.setFoldMarginColors(N2C(fold_margin_style.fgColor), N2C(fold_margin_style.bgColor))

        selection_style = theme.getGlobalStyle('Selected text colour')
        if selection_style:
            _editor.setSelectionBackgroundColor(N2C(selection_style.bgColor))

        for style in list(theme.globalStyle.values()):

            if style.styleID > 0:
                _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE, style.styleID, N2S(style.fgColor))

                _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK, style.styleID, N2S(style.bgColor))

                _editor.SendScintilla(QsciScintilla.SCI_STYLESETBOLD, style.styleID, style.fontStyle)

        fold_style = theme.getGlobalStyle('Fold')
        if fold_style:

            fold_marker_enums = [QsciScintilla.SC_MARKNUM_FOLDEREND, QsciScintilla.SC_MARKNUM_FOLDEROPENMID,
                               QsciScintilla.SC_MARKNUM_FOLDERMIDTAIL,
                               QsciScintilla.SC_MARKNUM_FOLDERTAIL, QsciScintilla.SC_MARKNUM_FOLDERSUB,
                               QsciScintilla.SC_MARKNUM_FOLDER, QsciScintilla.SC_MARKNUM_FOLDEROPEN]

            # to be consistent with notepad++ we switch bgColor with fgColor to style fold markers    
            for fold_marker in fold_marker_enums:
                # notice, marker foreground is styled using bgColor
                _editor.SendScintilla(QsciScintilla.SCI_MARKERSETFORE, fold_marker, N2S(fold_style.bgColor))

                # notice, marker background is styled using fgColor
                _editor.SendScintilla(QsciScintilla.SCI_MARKERSETBACK, fold_marker, N2S(fold_style.fgColor))


    def getStyleFromTheme(self, _styleName, _themeName):

        N2C = self.npStrToQColor
        N2S = self.npStrToSciColor

        try:
            theme = self.themeDict[_themeName]
        except LookupError:
            print(type(_themeName))
            print('Could not find theme: ' + _themeName + ' in ThemeManager')
            print('got these themes=', list(self.themeDict.keys()))
            return None

        style = theme.getGlobalStyle(_styleName)

        if style:
            return style
        else:
            return None

    def applyThemeToEditor(self, _themeName, _editor):

        N2C = self.npStrToQColor

        N2S = self.npStrToSciColor

        try:
            theme = self.themeDict[_themeName]
        except LookupError:

            print(type(_themeName))
            print('111Could not find theme: ' + _themeName + ' in ThemeManager')
            print('got these themes=', list(self.themeDict.keys()))
            return

        lexer = _editor.lexer()

        if not lexer:

            self.applyGlobalStyleItems(theme, _editor)
            return

        lexer_language = str(lexer.language())

        try:
            npp_styler_language_name = self.sciltillaLexerToNppTheme[lexer_language.lower()]
        except LookupError:
            npp_styler_language_name = lexer_language.lower()

        lexer_style = theme.getLexerStyle(npp_styler_language_name)

        if not lexer_style: return

        # applying global styles

        self.applyGlobalStyleItems(theme, _editor)

        for word_style in lexer_style.wordStyles:
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE, word_style.styleID, N2S(word_style.fgColor))
            _editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK, word_style.styleID, N2S(word_style.bgColor))
