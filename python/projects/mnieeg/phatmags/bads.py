from dataclasses import dataclass, field
import mne

@dataclass
class BadsAnnotations:
    Alpha: mne.Annotations = None
    Beta: mne.Annotations = None
    MonoMu: mne.Annotations = None
    MonoMu_Radial: mne.Annotations = None
    PowerInformed: mne.Annotations = None

@dataclass
class BadsSubject:
    Alpha: list[int] = field(default_factory=list)
    Beta: list[int] = field(default_factory=list)
    MonoMu: list[int] = field(default_factory=list)
    MonoMu_Radial: list[int] = field(default_factory=list)
    PowerInformed: list[int] = field(default_factory=list)
    Annotations: BadsAnnotations = field(default_factory=BadsAnnotations)
    exclude: bool = False
    exclude_reason: str = ''

class BadsPhatmags:
    X01751 = BadsSubject()

    # bad contact?
    X01847 = BadsSubject(Alpha=[41, 58],
                         MonoMu=[41, 58])

    # fairly flat
    X03754 = BadsSubject(PowerInformed=[48],
                         Annotations=BadsAnnotations(
                             PowerInformed=mne.Annotations(
                                 [260, 317], [20, 3], 'BAD_SEGMENT')))

    X04005 = BadsSubject()

    # 50, 51, 52 and 63, 64, 65 do not look good (occipital)
    X04441 = BadsSubject()

    # Lots of eyeblinks and other artifacts
    X09278 = BadsSubject(Annotations=BadsAnnotations(
                             MonoMu_Radial=mne.Annotations(
                                 68, 3, 'BAD_SEGMENT')))

    # 17, 18; 31, 32; 41, 42; 55, 56 cluster
    # Perhaps myogenic / muscle artifact
    X10415 = BadsSubject()

    X11547 = BadsSubject()

    X12169 = BadsSubject(Alpha=[68])

    X12695 = BadsSubject(Annotations=BadsAnnotations(
                            MonoMu_Radial=mne.Annotations(
                                32, 2, 'BAD_SEGMENT')))

    # high frequency content in 21-23, 41, 43-47, 55-58
    X13763 = BadsSubject()

    # 47, 55, 67 # bad contact?
    X20469 = BadsSubject(Annotations=BadsAnnotations(
                            PowerInformed=mne.Annotations(
                                [0, 215], [30, 7], 'BAD_SEGMENT')))

    X24987 = BadsSubject(Annotations=BadsAnnotations(
                            MonoMu=mne.Annotations(
                                [37, 173], [5, 12], 'BAD_SEGMENT')))

    # Lots of muscle artifacts - consider excluding?
    X27160 = BadsSubject(Alpha=[6], Annotations=BadsAnnotations(
                             MonoMu=mne.Annotations(
                                 [53, 172], [5, 8], 'BAD_SEGMENT')))

    # 21, 22, 23, 34, 45, 46, 67
    X27528 = BadsSubject()

    # Lots of muscle artifacts - consider excluding
    X34245 = BadsSubject(Annotations=BadsAnnotations(
                            PowerInformed=mne.Annotations(
                                [291, 327], [30, 40], 'BAD_SEGMENT')))

    X36945 = BadsSubject()

    # 47, 61 # bad contact?
    X45466 = BadsSubject()

    # bad contact?
    # bad contact? 45, 46, 47, 56, 57 not very good
    X48822 = BadsSubject(MonoMu_Radial=[47], PowerInformed=[47],
                         Annotations=BadsAnnotations(
                             MonoMu_Radial=mne.Annotations(
                                 [0, 142, 184], [5, 7, 5], 'BAD_SEGMENT'),
                             PowerInformed=mne.Annotations(
                                 [253], [5], 'BAD_SEGMENT')))

    # 23, 46, 59 # bad contact?
    # Lots of muscle artifacts

    X49038 = BadsSubject(Annotations=BadsAnnotations(
                            MonoMu=mne.Annotations(
                                107, 1, 'BAD_SEGMENT')))

    # Lots of eyeblinks both conditions
    # Alpha: 62, 63, 64, 65, 66 # bad contact?
    X60110 = BadsSubject(Alpha=[2])

    X62231 = BadsSubject(MonoMu=[6], Annotations=BadsAnnotations(
                            MonoMu=mne.Annotations(
                                [126, 182], [3, 3], 'BAD_SEGMENT')))

    # 47-8, 54-5 # bad contact?
    X64676 = BadsSubject()

    # bad contact?
    X69872 = BadsSubject(Alpha=[55], Annotations=BadsAnnotations(
                            MonoMu=mne.Annotations(
                                [110], [3], 'BAD_SEGMENT')))

    # fairly flat
    X70329 = BadsSubject(PowerInformed=[48], Annotations=BadsAnnotations(
                             MonoMu_Radial=mne.Annotations(
                                 [117, 137], [5, 3], 'BAD_SEGMENT'),
                             PowerInformed=mne.Annotations(
                                 [70, 207, 307], [3, 2, 7], 'BAD_SEGMENT')
                            )
                        )

    X73586 = BadsSubject()

    X73659 = BadsSubject(exclude=True,
                         exclude_reason='only 8 s data available')

    # MonoMu : Lots of eyeblinks
    X75729 = BadsSubject(Annotations=BadsAnnotations(
                            MonoMu=mne.Annotations(
                                [17, 33], [3, 4], 'BAD_SEGMENT')))

    # Lots of muscle artifacts
    X76349 = BadsSubject()

    X79863 = BadsSubject(exclude=True,
                         exclude_reason='only 45 s data available')

    # Lots of muscle artifacts
    X81781 = BadsSubject(Annotations=BadsAnnotations(
                            MonoMu_Radial=mne.Annotations(
                                0, 50, 'BAD_SEGMENT')))

    X84315 = BadsSubject()

    X84907 = BadsSubject(Annotations=BadsAnnotations(
                            MonoMu=mne.Annotations(0, 20, 'BAD_SEGMENT')))

    # 57, 63 bad contact?
    X87100 = BadsSubject(Annotations=BadsAnnotations(
                            MonoMu_Radial=mne.Annotations(
                                [16], [3], 'BAD_SEGMENT'),
                            PowerInformed=mne.Annotations(
                                [8, 297, 357, 390, 435, 480],
                                [14, 10, 10, 22, 10, 10], 'BAD_SEGMENT')))

    # bad contact?
    # PowerInformed : 44, 45, 57, 58 bad contact?
    X99472 = BadsSubject(MonoMu=[45, 57], Annotations=BadsAnnotations(
                            MonoMu=mne.Annotations(
                                [0, 30, 62, 182], [10, 10, 4, 3],
                                'BAD_SEGMENT'),
                            PowerInformed=mne.Annotations(
                                [0, 122, 172], [80, 4, 19], 'BAD_SEGMENT')))

"""
phatmags_bads = {
    'X01751' : {
        'Alpha' : [], # 47 start and end
        'MonoMu' : [],
        'exclude' : False
    },
    'X01847' : {
        'Alpha' : [41, 58], # bad contact?
        'MonoMu' : [41, 58],  # bad contact?
        'exclude' : False
    },
    'X03754' : {
        'PowerInformed' : [48], # fairly flat
        'annotations' : {
            'PowerInformed' : mne.Annotations([260, 317], [20, 3],'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X04005' : {
        'MonoMu' : [],
        'exclude' : False
    },
    'X04441' : {
        # 50, 51, 52 and 63, 64, 65 do not look good (occipital)
        'MonoMu_Radial' : [],
        'exclude' : False
    },
    'X09278' : {
        # Lots of eyeblinks and other artifacts
        'MonoMu_Radial' : [],
        'annotations' : {
            'MonoMu_Radial' : mne.Annotations(68, 3, 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X10415' : {
        # 17, 18; 31, 32; 41, 42; 55, 56 cluster
        # Perhaps myogenic / muscle artifact
        'MonoMu_Radial' : [],
        'exclude' : False
    },
    'X11547' : {
        'PowerInformed' : [],
        'exclude' : False
    },
    'X12169' : {
        'Alpha' : [68],
        'exclude' : False
    },
    'X12695' : {
        'MonoMu_Radial' : [],
        'annotations' : {
            'MonoMu_Radial' : mne.Annotations(32, 2, 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X13763' : {
        # high frequency content in 21-23, 41, 43-47, 55-58
        'MonoMu_Radial' : [],
        'exclude' : False
    },
    'X20469' : {
        'PowerInformed' : [], # 47, 55, 67 # bad contact?
        'annotations' : {
            'PowerInformed' : mne.Annotations([0, 215], [30, 7], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X24987' : {
        'MonoMu' : [],
        'annotations' : {
            'MonoMu' : mne.Annotations([37, 173], [5, 12], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X27160' : {
        # Lots of muscle artifacts - consider excluding?
        'Alpha' : [6],
        'MonoMu' : [],
        'annotations' : {
            'MonoMu' : mne.Annotations([53, 172], [5, 8], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X27528' : {
        'PowerInformed' : [], # 21, 22, 23, 34, 45, 46, 67
        'exclude' : False
    },
    'X34245' : {
        # Lots of muscle artifacts - consider excluding
        'PowerInformed' : [],
        'annotations' : {
            'PowerInformed' : mne.Annotations([291, 327], [30, 40], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X36945' : {
        'MonoMu_Radial' : [],
        'exclude' : False
    },
    'X45466' : {
        'MonoMu' : [], # 47, 61 # bad contact?
        'exclude' : False
    },
    'X48822' : {
        # bad first ~20 s
        'MonoMu_Radial' : [47], # bad contact?
        'PowerInformed' : [47], # bad contact? 45, 46, 47, 56, 57 not very good
        'annotations' : {
            'MonoMu_Radial' : mne.Annotations([0, 142, 184], [5, 7, 5], 'BAD_SEGMENT'),
            'PowerInformed' : mne.Annotations([253], [5], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X49038' : {
        'Alpha' : [], # 23, 46, 59 # bad contact?
        'MonoMu' : [], # Lots of muscle artifacts
        'annotations' : {
            'MonoMu' : mne.Annotations(107, 1, 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X60110' : {
        # Lots of eyeblinks both conditions
        'Alpha' : [2], # 62, 63, 64, 65, 66 # bad contact?
        'MonoMu' : [],
        'exclude' : False
    },
    'X62231' : {
        'MonoMu' : [6],
        'annotations' : {
            'MonoMu' : mne.Annotations([126, 182], [3, 3], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X64676' : {
        'MonoMu' : [], # 47-8, 54-5 # bad contact?
        'exclude' : False
    },
    'X69872' : {
        'Alpha' : [55], # bad contact?
        'MonoMu' : [],
        'annotations' : {
            'MonoMu' : mne.Annotations([110], [3], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X70329' : {
        'MonoMu_Radial' : [],
        'PowerInformed' : [48], # fairly flat
        'annotations' : {
            'MonoMu_Radial' : mne.Annotations([117, 137], [5, 3], 'BAD_SEGMENT'),
            'PowerInformed' : mne.Annotations([70, 207, 307], [3, 2, 7], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X73586' : {
        'MonoMu_Radial' : [],
        'PowerInformed' : [],
        'exclude' : False
    },
    'X73659' : {
        'PowerInformed' : [], # Only 8 s
        'exclude' : True,
        'exclude_reason' : 'only 8 s data available'
    },
    'X75729' : {
        'MonoMu' : [], # Lots of eyeblinks
        'annotations' : {
            'MonoMu' : mne.Annotations([17, 33], [3, 4], 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X76349' : {
        # Lots of muscle artifacts
        'Alpha' : [],
        'MonoMu' : [],
        'exclude' : False
    },
    'X79863' : {
        'MonoMu_Radial' : [], # only 45 s, high frequency noise
        'exclude' : True,
        'exclude_reason' : 'only 45 s data available'
    },
    'X81781' : {
        # Lots of muscle artifacts
        'MonoMu_Radial' : [],
        'annotations' : {
            'MonoMu_Radial' : mne.Annotations(0, 50, 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X84315' : {
        'MonoMu_Radial' : [],
        'exclude' : False
    },
    'X84907' : {
        'MonoMu' : [],
        'annotations' : {
            'MonoMu' : mne.Annotations(0, 20, 'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X87100' : {
        'MonoMu_Radial' : [],
        'PowerInformed' : [], # 57, 63 # bad contact?
        'annotations' : {
            'MonoMu_Radial' : mne.Annotations([16], [3], 'BAD_SEGMENT'),
            'PowerInformed' : mne.Annotations([8, 297, 357, 390, 435, 480],
                                              [14, 10, 10, 22, 10, 10],
                                              'BAD_SEGMENT')
        },
        'exclude' : False
    },
    'X99472' : {
        'MonoMu' : [45, 57], # bad contact?
        'PowerInformed' : [], # 44, 45, 57, 58 # bad contact?
        'annotations' : {
            'MonoMu' : mne.Annotations([0, 30, 62, 182], [10, 10, 4, 3],
                                       'BAD_SEGMENT'),
            'PowerInformed' : mne.Annotations([0, 122, 172], [80, 4, 19],
                                              'BAD_SEGMENT')
        },'exclude' : False
    }
}
"""