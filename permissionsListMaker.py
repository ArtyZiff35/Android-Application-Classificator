import codecs
from ehp import *
from bs4 import BeautifulSoup
import re


permissionsList = []

filepath = './supportFiles/androidPermissionsSource.html'
outpath = './fullPermissionsListUpdated.txt'

with codecs.open(filepath, encoding="utf8") as fp:
    line = fp.readline()
    cnt = 1
    while line:
        if "api-name" in line:
            perm = re.search('>.*<' , line)
            if perm:
                permission = str(perm.group(0)).replace(">", "")
                permission = permission.replace("<","")
                print(permission)
                permissionsList.append(permission)
                cnt += 1
        line = fp.readline()
print(cnt)


with open(outpath, "w") as fp:
    for perm in permissionsList:
        fp.write("%s\n" % perm)

print("\n\nTask Done")