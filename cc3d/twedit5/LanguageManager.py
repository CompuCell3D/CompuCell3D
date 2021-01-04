from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5 import twedit
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, errMsg, dbgMsg

# first determine where APIs are located

# initial guess is in the "APIs" subrirestory of the directory which holds Configuration.py

# tweditRootPath = os.path.dirname(Configuration.__file__)
tweditRootPath = os.path.dirname(os.path.dirname(twedit.__file__))
apisPath = os.path.join(tweditRootPath, "APIs")

# check if it exists

if not os.path.exists(apisPath):
    # when packaging on Windows with pyinstaller the path to executable is accesible via
    # sys.executable as Python is bundled with the distribution

    # os.path.dirname(Configuration.__file__) returned by pyInstaller will not work without some
    # modifications so it is best tu use os.path.dirname(sys.executable) approach

    tweditRootPath = os.path.dirname(sys.executable)

    apisPath = os.path.join(tweditRootPath, "APIs")


LANGS = {
    "Bash": ["QsciLexerBash", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Batch": ["QsciLexerBatch", "REM ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "C": ["QsciLexerCPP", "// ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "C#": ["QsciLexerCSharp", "// ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "CMake": ["QsciLexerCMake", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "CSS": ["QsciLexerCSS", "/* ", " */", 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "D": ["QsciLexerD", "// ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Diff": ["QsciLexerDiff", None, None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Fortran": ["QsciLexerFortran", "! ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Fortran77": ["QsciLexerFortran77", "c ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "HTML": ["QsciLexerHTML", "<!-- ", " -->", 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "IDL": ["QsciLexerIDL", "; ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Java": ["QsciLexerJava", "// ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "JavaScript": ["QsciLexerJavaScript", "// ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Lua": ["QsciLexerLua", "-- ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Makefile": ["QsciLexerMakefile", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Matlab":  ["QsciLexerMatlab", "% ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Octave": ["QsciLexerOctave", "% ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Pascal": ["QsciLexerPascal", "{ ", " }", 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Perl": ["QsciLexerPerl", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "PostScript": ["QsciLexerPostScript", "% ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Properties": ["QsciLexerProperties", "; ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Python": ["QsciLexerPython", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "POV": ["QsciLexerPOV", "// ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Spice": ["QsciLexerSpice", "* ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "SQL": ["QsciLexerSQL", "/* ", " */", 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Ruby": ["QsciLexerRuby", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "TCL": ["QsciLexerTCL", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "Verilog": ["QsciLexerVerilog", "// ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "VHDL": ["QsciLexerVHDL", "-- ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "TeX": ["QsciLexerTeX", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "XML": ["QsciLexerXML", "<!-- ", " -->", 1, 5, QsciScintilla.SCWS_INVISIBLE],
    "YML": ["QsciLexerYAML", "# ", None, 1, 5, QsciScintilla.SCWS_INVISIBLE]
}
"""Supported languages and their lexers and lexer properties"""

APIS = {
    "QsciLexerCPP": os.path.abspath(os.path.join(apisPath, "cplusplus.api")),
    "QsciLexerCSharp": os.path.abspath(os.path.join(apisPath, "csharp.api")),
    "QsciLexerCSS": os.path.abspath(os.path.join(apisPath, "css.api")),
    "QsciLexerHTML": os.path.abspath(os.path.join(apisPath, "html.api")),
    "QsciLexerJava": os.path.abspath(os.path.join(apisPath, "java.api")),
    "QsciLexerJavaScript": os.path.abspath(os.path.join(apisPath, "javascript.api")),
    "QsciLexerPearl": os.path.abspath(os.path.join(apisPath, "perl.api")),
    "QsciLexerRuby": os.path.abspath(os.path.join(apisPath, "ruby.api"))
}
"""Built-in apis; automatically loaded"""

if isfile(join(tweditRootPath, "APIs", f"Python-{sys.version_info.major}.{sys.version_info.minor}.api")):
    APIS["QsciLexerPython"] = join(tweditRootPath, "APIs",
                                   f"Python-{sys.version_info.major}.{sys.version_info.minor}.api")
else:
    APIS["QsciLexerPython"] = join(apisPath, "python.api")


class LanguageManager:

    def __init__(self, _editorWindow):

        self.editorWindow = _editorWindow

        self.actionDict = {}

        # it maps lexer class name to lexer class object e.g. "QsciLexerCPP":QsciLexerCPP()
        self.lexerObjectDict = {}

        self.languageMapper = QSignalMapper(self.editorWindow)

        # self.editorWindow.connect(self.languageMapper,SIGNAL("mapped(const QString&)"),  self.selectLexer)

        self.languageMapper.mapped[str].connect(self.selectLexer)

        self.actionChecked = None

        self.importAllAvailableLexers()

        self.apiDict = {}

        self.api_file_dict = {v[0]: [] for v in LANGS.values()}
        """Dictionary containing .api files for autocompletion"""

        self.installAutocompletionAPIs()

        # format [lexer,begin comment, end comment, brace matching (0- nor matching, 1 matching), codeFolding]

        # e.g. "Bash":[QsciLexerBash(),"# ",None,1,5,QsciScintilla.SCWS_INVISIBLE],

        self.languageLexerDictionary = {}

        self.lexerLanguageDictionary = {}

        [self.addLanguageLexerDictionaryEntry(k, v.copy()) for k, v in LANGS.items()]

        self.lDict = {"XML": QsciLexerXML()}

    def importAllAvailableLexers(self):

        for lexerName in [v[0] for v in LANGS.values()]:

            try:
                exec("from PyQt5.Qsci import " + lexerName + "\n")

                self.lexerObjectDict[lexerName] = eval(lexerName + "()")

            except ImportError:

                pass

    def loadSingleAPI(self, _lexerName, _apiName):

        try:

            if _apiName not in self.api_file_dict[_lexerName]:

                self.api_file_dict[_lexerName].append(_apiName)

        except KeyError:

            return

    def installAutocompletionAPIs(self):

        dbgMsg("apisPath=", os.path.abspath(apisPath))

        [self.loadSingleAPI(k, v) for k, v in APIS.items()]

    def addLanguageLexerDictionaryEntry(self, _languageName, _languagePropertiesData):

        """

            _languagePropertiesData=["QsciLexerTeX","# ",None,1,5,QsciScintilla.SCWS_INVISIBLE]

        """

        try:

            _languagePropertiesData[0] = self.lexerObjectDict[_languagePropertiesData[0]]

            self.languageLexerDictionary[_languageName] = _languagePropertiesData

            self.lexerLanguageDictionary[self.languageLexerDictionary[_languageName][0]] = _languageName

        except (KeyError, IndexError):
            pass

    def createActions(self):

        keys = list(self.languageLexerDictionary.keys())

        keys.sort()

        for key in keys:
            action = self.editorWindow.languageMenu.addAction(key)

            self.actionDict[key] = action

            action.setCheckable(True)

            # self.editorWindow.connect(action, SIGNAL("triggered()"), self.languageMapper, SLOT("map()"))

            action.triggered.connect(self.languageMapper.map)

            self.languageMapper.setMapping(action, key)

            # self.actionDict[key]=QtGui.QAction(key, self, shortcut="",

            # statusTip=key, triggered=self.increaseIndent)

    def finalize_apis(self):
        """
        All APIs loaded internally and externally are finalized

        :return: None
        """

        for lexer_name, api_files in self.api_file_dict.items():

            api = QsciAPIs(self.lexerObjectDict[lexer_name])

            [api.load(af) for af in api_files]

            api.prepare()

            self.lexerObjectDict[lexer_name].setAPIs(api)

            self.apiDict[lexer_name] = api

    def selectLexerBasedOnLexerObject(self, _lexerObj):

        try:

            languageName = self.lexerLanguageDictionary[_lexerObj]

            self.updateLexerActions(languageName)

        except (KeyError, IndexError):

            self.updateLexerActions('')  # no lexer is selected

            pass

    def updateLexerActions(self, _language):

        # in case language could not be figured out we uncheck current action (if one is checked) in the Language menu
        if _language == '':

            if self.actionChecked:
                self.actionChecked.setChecked(False)

            return

        try:

            if self.actionChecked:

                self.actionChecked.setChecked(False)

                self.actionDict[str(_language)].setChecked(True)

                self.actionChecked = self.actionDict[str(_language)]

            else:

                self.actionDict[str(_language)].setChecked(True)

                self.actionChecked = self.actionDict[str(_language)]

        except (KeyError, IndexError):

            pass

    def selectLexer(self, _language):

        try:

            if self.actionChecked:

                self.actionChecked.setChecked(False)

                self.actionDict[str(_language)].setChecked(True)

                self.actionChecked = self.actionDict[str(_language)]

            else:

                self.actionDict[str(_language)].setChecked(True)

                self.actionChecked = self.actionDict[str(_language)]

            # setting Lexer
            editor = self.editorWindow.getActiveEditor()
            editor.setLexer(self.languageLexerDictionary[str(_language)][0])
            editor.lexer().setFont(self.editorWindow.baseFont)

            self.editorWindow.commentStyleDict[editor] = self.languageLexerDictionary[str(_language)][1:3]

            # after changing lexer we have to set its properties incluging those defined by themes
            self.editorWindow.setEditorProperties(editor)

        except (KeyError, IndexError):
            pass
