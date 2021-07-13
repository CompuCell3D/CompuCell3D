import re

from collections import namedtuple

SnippetTuple = namedtuple('SnippetTuple', 'snippet_text suggested_indent')


class SnippetMenuParser(object):

    def __init__(self):

        self.snippetMenu = {}

        self.currentMenu = None

        self.currentSubmenu = None

        self.currentSnippet = None

        self.currentSuggestedIndent = None

        self.menuRegex = re.compile('^[=]*[\s]*#[\s]*@Menu@([\s\S]*)$')

        # self.submenuRegex = re.compile('^[-]*[\s]*#[\s]*@Submenu@([\s\S]*)$')

        # self.submenuRegex = re.compile('^[-]*([i\d]*)[\s]*#[\s]*@Submenu@([\s\S]*)$')

        # self.submenuRegex = re.compile('^[-]*([\s]*|[[i\d]*\s*])#[\s]*@Submenu@([\s\S]*)$')

        self.submenuRegex = re.compile('^[-]*([i\d]*)[\s]*#[\s]*@Submenu@([\s\S]*)$')

    def initialize(self):

        self.snippetMenu = {}

        self.currentMenu = None

        self.currentSubmenu = None

        self.currentSnippet = None

    def getSnippetMenuDict(self):

        return self.snippetMenu

    def findToken(self, _line, _regex, group_idx=0):

        line = _line.rstrip()

        for m in _regex.finditer(line):
            tokenGroup = m.groups()

            # print 'menu token Group=',tokenGroup

            return tokenGroup[group_idx]

        return None

    def writeSnippet(self):

        if self.currentSnippet and self.currentMenu and self.currentSubmenu:

            self.currentMenu[self.currentSubmenu] = self.currentSnippet

            if self.currentSuggestedIndent:

                self.currentMenu[self.currentSubmenu] = SnippetTuple(self.currentSnippet, self.currentSuggestedIndent)

            else:

                self.currentMenu[self.currentSubmenu] = SnippetTuple(self.currentSnippet, -1)

    def readSnippetMenu(self, _fileName):

        file = open(_fileName)

        readyToAddSnippet = False

        for line in file:

            menuName = self.findToken(line, self.menuRegex)

            # print 'menuName=',menuName

            if menuName:
                self.writeSnippet()

                readyToAddSnippet = False

                self.snippetMenu[menuName] = {}

                self.currentMenu = self.snippetMenu[menuName]

                continue

            submenuName = self.findToken(line, self.submenuRegex, group_idx=1)

            suggested_indent = self.findToken(line, self.submenuRegex, group_idx=0)

            # print 'suggested_indent=',suggested_indent

            if submenuName is not None:

                submenuName = submenuName.strip()

                if suggested_indent:
                    self.currentSuggestedIndent = int(suggested_indent[1:])

            # print 'submenuName=',submenuName

            if submenuName:

                # writing previous snippet ()

                self.writeSnippet()

                self.currentSubmenu = submenuName

                self.currentMenu[submenuName] = ''

                self.currentSnippet = ''

                readyToAddSnippet = True

                if suggested_indent:

                    self.currentSuggestedIndent = int(suggested_indent[1:])

                else:

                    self.currentSuggestedIndent = -1

                continue

            if readyToAddSnippet: self.currentSnippet += line

        self.writeSnippet()

        file.close()

    # if __name__=='__main__':

    # psmp = PythonSnippetMenuParser()

    # psmp.readSnippetMenu('Snippets.py.template')

    # print 'snippet menu dict = ',psmp.getSnippetMenuDict()
