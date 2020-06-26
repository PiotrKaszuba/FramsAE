from subprocess import Popen, PIPE


pr= Popen('C:/Users/Piotr/Desktop/Framsticks50rc14\\frams.exe -V', stdout=PIPE, stderr=PIPE, stdin=PIPE)
print([line.decode('utf-8').rstrip() for line in iter(pr.stdout.readlines())])