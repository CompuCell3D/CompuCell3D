import re


class SnippetMenuParser(object):
    def __init__(self):
    
        
        self.snippetMenu={}            
        self.currentMenu=None    
        self.currentSubmenu=None
        self.currentSnippet=None
        
        self.menuRegex=re.compile('^[=]*[\s]*#@Menu@([\s\S]*)$')
        self.submenuRegex=re.compile('^[-]*[\s]*#@Submenu@([\s\S]*)$')
        
    def initialize(self):
        self.snippetMenu={}            
        self.currentMenu=None    
        self.currentSubmenu=None
        self.currentSnippet=None
        
    def getSnippetMenuDict(self):
        return self.snippetMenu
        
    def findToken(self,_line,_regex,):
        line = _line.rstrip()
        for m in _regex.finditer(line):   
             tokenGroup=m.groups()
             # print 'menu token Group=',tokenGroup
             return tokenGroup[0]
             
        return None          
        
    def writeSnippet(self):
        if self.currentSnippet and self.currentMenu and self.currentSubmenu:
            self.currentMenu[self.currentSubmenu]=self.currentSnippet
            
    def readSnippetMenu(self,_fileName):
        file=open(_fileName)
        
        readyToAddSnippet=False
        for line in file:

            menuName=self.findToken(line,self.menuRegex)
            # print 'menuName=',menuName
            if menuName:
                self.writeSnippet()
                readyToAddSnippet=False
                self.snippetMenu[menuName]={}
                self.currentMenu=self.snippetMenu[menuName]
                continue

            submenuName=self.findToken(line,self.submenuRegex)        
            # print 'submenuName=',submenuName
            if submenuName:
            
                self.writeSnippet()
                
                
                self.currentSubmenu = submenuName       
                
                self.currentMenu[submenuName]=''
                self.currentSnippet=''
                readyToAddSnippet=True
                continue
                
            if readyToAddSnippet: self.currentSnippet+=line
            
        self.writeSnippet()        
        file.close()

# if __name__=='__main__':
        
    # psmp = PythonSnippetMenuParser()

    # psmp.readSnippetMenu('Snippets.py.template')

    # print 'snippet menu dict = ',psmp.getSnippetMenuDict()

