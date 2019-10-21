class SQList:
    def __init__(self, lis=None):
        self.r = lis

    def swap(self, i, j):
        """定义一个交换元素的方法，方便后面调用。"""
        temp = self.r[i]
        self.r[i] = self.r[j]
        self.r[j] = temp

    def selection_sort(self):
        """简单选择排序，时间复杂度O(n^2)"""
        lis = self.r
        length = len(self.r)
        for i in range(length):
            minimum = i
            for j in range(i+1, length):
                if lis[minimum] > lis[j]:
                    minimum = j
            if i != minimum:
                self.swap(i, minimum)

    def __str__(self):
        ret = ""
        for i in self.r:
            ret += " %s" %i
        return ret

if __name__ == "__main__":
    sqlist = SQList([5,2,3,9,7,1,8,6,4,0])
    sqlist.selection_sort()
    print(sqlist)