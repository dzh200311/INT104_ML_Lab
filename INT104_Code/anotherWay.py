def lcs(s1, s2):
    m = len(s1)
    n = len(s2)

    # 创建一个二维数组来保存子问题的解
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充二维数组，计算子问题的解
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 从二维数组的右下角开始回溯，构造最长公共子序列
    lcs_sequence = []
    i = m
    j = n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs_sequence.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 反转最长公共子序列
    lcs_sequence.reverse()

    return ''.join(lcs_sequence)

# 测试示例
s2 = "GAGT"
s1 = "AGACCT"

result = lcs(s1, s2)
print("最长公共子序列：", result)
