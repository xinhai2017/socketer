class SQList:
    def __init__(self, lis=None):
        self.r = lis

    def insert_sort(self):
        lis = self.r
        length = len(self.r)
        #下标从1开始
        for i in range(1, length):
            if lis[i] < lis[i-1]:
                temp = lis[i]
                j = i - 1
                while lis[j] > temp and j >= 0:
                    lis[j + 1] = lis[j]
                    j -= 1
                lis[j + 1] = temp

    def __str__(self):
        ret = ""
        for i in self.r:
            ret += " %s" %i
        return ret

if __name__=='__main__':
    sqlist = SQList([5,2,3,9,7,1,8,6,4,0])
    sqlist.insert_sort()
    print(sqlist)