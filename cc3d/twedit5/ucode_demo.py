import locale

import unicodedata



from os.path import *

import os

from _functools import partial



pref_encoding =  locale.getpreferredencoding()



def unicode_cnv_fcn(_str, encoding):

    return unicodedata.normalize('NFKD', _str).encode(encoding, 'ignore').strip()



unicode_cnv = partial(unicode_cnv_fcn,encoding=locale.getpreferredencoding())



# a = unicode('kro\xcc\x81l','utf-8')

# b = unicode('kr\xc3\xb3l','utf-8')

a = str('kro\xcc\x81l ',pref_encoding)

b = str('kr\xc3\xb3l',pref_encoding)



c = 'kr\xf3l'



print(a)

print(b)

print(c)



a_n = unicode_cnv(a)

b_n = unicode_cnv(b)

c_n = unicode_cnv(c)



# a_n = unicodedata.normalize('NFKD', a).encode('utf-8','ignore')

# b_n = unicodedata.normalize('NFKD', b).encode('utf-8','ignore')

# c_n = unicodedata.normalize('NFKD', c).encode('utf-8','ignore')



print('a_n=',a_n)

print('b_n=',b_n)

print('c_n=',c_n)



print(a_n==b_n)



print(c_n.strip()==b_n.strip())



the_dir_raw = os.listdir(expanduser('~'))[73].strip()

print('repr(the_dir_raw)=',repr(the_dir_raw))

the_dir = str(the_dir_raw,'utf-8')

print('the_dir repr', the_dir)



the_dir_conv = unicode_cnv(the_dir)

print('the_dir_conv=',the_dir_conv)

print('repr the_dir_conv=',repr(the_dir_conv))

print('the_dir=',the_dir)



print(the_dir_raw == a_n)





full_the_dir_raw = join('/Users/m',the_dir_raw)



print(isdir(full_the_dir_raw))

print(full_the_dir_raw)

# print the_dir.strip() == a_n.strip()

# the_dir =  unicode_cnv(unicode(os.listdir(expanduser('~'))[73],pref_encoding))

# full_dir = join(expanduser('~'),the_dir)

#

# full_dir_conv = join(expanduser('~'),a_n)



# print 'full_dir=',full_dir

# print 'full_dir_conv=',full_dir_conv

#

# print full_dir==full_dir_conv







# 'Kluft skrams infor pa federal electoral groe'



# a_dc = a.decode('utf-8')

# print a_dc