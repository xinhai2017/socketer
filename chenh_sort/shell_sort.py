class SQList:
    def __init__(self, lis=None):
        self.r = lis

    def shell_sort(self):
        """希尔排序"""
        lis = self.r
        length = len(lis)
        increment = len(lis)
        while increment > 1:
            increment = int(increment/3) + 1
            for i in range(increment+1, length):
                if lis[i] < lis[i - increment]:
                    temp = lis[i]
                    j = i - increment
                    while j >= 0 and temp < lis[j]:
                        lis[j + increment] = lis[j]
                        j -= increment
                    lis[j + increment] = temp

    def __str__(self):
        ret = ""
        for i in self.r:
            ret += " %s" %i
        return ret

if __name__=="__main__":
    sqlist = SQList([5,2,3,9,7,1,8,6,4,0])
    sqlist.shell_sort()
    print(sqlist)
