dataset_info = dict(
    dataset_name='camaron',
    paper_info=dict(
        author='IDN',
        title='IDN',
        container='Nature methods',
        year='2025',
        homepage='IDN',
    ),
    keypoint_info={
        0:
        dict(name='part0', id=0, color=[255, 255, 255], type='', swap=''),
        1:
        dict(name='part1', id=1, color=[255, 255, 255], type='', swap=''),
        2:
        dict(name='part2', id=2, color=[255, 255, 255], type='', swap=''),
        3:
        dict(name='part3', id=3, color=[255, 255, 255], type='', swap=''),
        4:
        dict(name='part4', id=4, color=[255, 255, 255], type='', swap=''),
        5:
        dict(name='part5', id=5, color=[255, 255, 255], type='', swap=''),
        6:
        dict(name='part6', id=6, color=[255, 255, 255], type='', swap=''),
        7:
        dict(name='part7', id=7, color=[255, 255, 255], type='', swap=''),
        8:
        dict(name='part8', id=8, color=[255, 255, 255], type='', swap=''),
        9:
        dict(name='part9', id=9, color=[255, 255, 255], type='', swap=''),
        10:
        dict(name='part10', id=10, color=[255, 255, 255], type='', swap=''),
        11:
        dict(name='part11', id=11, color=[255, 255, 255], type='', swap=''),
        12:
        dict(name='part12', id=12, color=[255, 255, 255], type='', swap=''),
        13:
        dict(name='part13', id=13, color=[255, 255, 255], type='', swap=''),
        14:
        dict(name='part14', id=14, color=[255, 255, 255], type='', swap=''),
        15:
        dict(name='part15', id=15, color=[255, 255, 255], type='', swap=''),
        16:
        dict(name='part16', id=16, color=[255, 255, 255], type='', swap=''),
        17:
        dict(name='part17', id=17, color=[255, 255, 255], type='', swap=''),
        18:
        dict(name='part18', id=18, color=[255, 255, 255], type='', swap=''),
        19:
        dict(name='part19', id=19, color=[255, 255, 255], type='', swap=''),
        20:
        dict(name='part20', id=20, color=[255, 255, 255], type='', swap=''),
        21:
        dict(name='part21', id=21, color=[255, 255, 255], type='', swap=''),
        22:
        dict(name='part22', id=22, color=[255, 255, 255], type='', swap=''),
        23:
        dict(name='part23', id=23, color=[255, 255, 255], type='', swap=''),
        24:
        dict(name='part24', id=24, color=[255, 255, 255], type='', swap=''),
        25:
        dict(name='part25', id=25, color=[255, 255, 255], type='', swap=''),
        26:
        dict(name='part26', id=26, color=[255, 255, 255], type='', swap=''),
        27:
        dict(name='part27', id=27, color=[255, 255, 255], type='', swap=''),
        28:
        dict(name='part28', id=28, color=[255, 255, 255], type='', swap=''),
        29:
        dict(name='part29', id=29, color=[255, 255, 255], type='', swap=''),
        30:
        dict(name='part30', id=30, color=[255, 255, 255], type='', swap=''),
        31:
        dict(name='part31', id=31, color=[255, 255, 255], type='', swap=''),
        32:
        dict(name='part32', id=32, color=[255, 255, 255], type='', swap=''),
        33:
        dict(name='part33', id=33, color=[255, 255, 255], type='', swap=''),
        34:
        dict(name='part34', id=34, color=[255, 255, 255], type='', swap=''),
        35:
        dict(name='part35', id=35, color=[255, 255, 255], type='', swap=''),
        36:
        dict(name='part36', id=36, color=[255, 255, 255], type='', swap=''),


    },
    skeleton_info={
        0: dict(link=('part0', 'part1'), id=0, color=[255, 255, 255]),
        1: dict(link=('part1', 'part2'), id=1, color=[255, 255, 255]),
        2: dict(link=('part2', 'part3'), id=2, color=[255, 255, 255]),
        3: dict(link=('part3', 'part4'), id=3, color=[255, 255, 255]),
        4: dict(link=('part4', 'part5'), id=4, color=[255, 255, 255]),
        5: dict(link=('part5', 'part6'), id=5, color=[255, 255, 255]),
        6: dict(link=('part6', 'part7'), id=6, color=[255, 255, 255]),
        7: dict(link=('part7', 'part8'), id=7, color=[255, 255, 255]),
        8: dict(link=('part9', 'part10'), id=8, color=[255, 255, 255]),
        9: dict(link=('part11', 'part12'), id=9, color=[255, 255, 255]),
        10: dict(link=('part13', 'part14'), id=10, color=[255, 255, 255]),
        11: dict(link=('part15', 'part16'), id=11, color=[255, 255, 255]),
        12: dict(link=('part17', 'part18'), id=12, color=[255, 255, 255]),
        13: dict(link=('part19', 'part20'), id=13, color=[255, 255, 255]),
        14: dict(link=('part21', 'part22'), id=14, color=[255, 255, 255]),
        15: dict(link=('part23', 'part24'), id=15, color=[255, 255, 255]),
        16: dict(link=('part25', 'part26'), id=16, color=[255, 255, 255]),
        17: dict(link=('part27', 'part28'), id=17, color=[255, 255, 255]),
        18: dict(link=('part29', 'part30'), id=18, color=[255, 255, 255]),
        19: dict(link=('part31', 'part32'), id=19, color=[255, 255, 255]),
        20: dict(link=('part33', 'part34'), id=20, color=[255, 255, 255]),
        21: dict(link=('part35', 'part36'), id=21, color=[255, 255, 255]),
    },
    joint_weights=[1.] * 37,
    sigmas=[.026, .026, .026, .025, .025,  .025, .035, .035, .035, .079, .079, .079, .072, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089, .079, .079, .079, .072, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089, ])
