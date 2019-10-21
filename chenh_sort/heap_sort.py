class SQList:
    def __init__(self, lis=None):
        self.r = lis

    def swap(self, i, j):
        """定义一个交换元素的方法，方便后面调用"""
        temp = self.r[i]
        self.r[i] = self.r[j]
        self.r[j] = temp

    def heap_sort(self):
        length = len(self.r)
        i = int(length/2)
        # 将原始序列构成一个大顶堆
        # 遍历从中间开始，到0结束，其实这些是堆的分支节点
        while i >= 0:
            self.heap_adjust(i, length-1)
            i -= 1
        # 逆序遍历整个序列，不断取出根结点的值，完成实际的排序。
        j = length - 1
        while j > 0:
            # 将当前根结点，也就是列表最开头，下表为0的值，交换到最后面j处
            self.swap(0, j)
            # 将发生变化的序列重新构造成大顶堆
            self.heap_adjust(0, j-1)
            j -= 1

    def heap_adjust(self, s, m):
        """核心的大顶堆构造方法，维持序列的堆结构。"""
        lis = self.r
        temp = lis[s]
        i = 2*s
        while i <= m:
            if i < m and lis[i] < lis[i+1]:
                i += 1
            if temp >= lis[i]:
                break
            lis[s] = lis[i]
            s = i
            i *= 2
        lis[s] = temp

    def __str__(self):
        ret = ""
        for i in self.r:
            ret += " %s" % i
        return ret

if __name__ == '__main__':
    sqlist = SQList([5,2,3,9,7,1,8,6,4,0])
    sqlist.heap_sort()
    print(sqlist)
