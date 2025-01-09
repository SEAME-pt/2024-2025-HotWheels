/****************************************************************************
** Meta object code from reading C++ file 'FakeSimulation.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../../app/CarCluster/includes/FakeSimulation.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'FakeSimulation.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_FakeSimulation_t {
    QByteArrayData data[8];
    char stringdata0[91];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_FakeSimulation_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_FakeSimulation_t qt_meta_stringdata_FakeSimulation = {
    {
QT_MOC_LITERAL(0, 0, 14), // "FakeSimulation"
QT_MOC_LITERAL(1, 15, 12), // "speedUpdated"
QT_MOC_LITERAL(2, 28, 0), // ""
QT_MOC_LITERAL(3, 29, 8), // "newSpeed"
QT_MOC_LITERAL(4, 38, 10), // "rpmUpdated"
QT_MOC_LITERAL(5, 49, 6), // "newRpm"
QT_MOC_LITERAL(6, 56, 18), // "simulationFinished"
QT_MOC_LITERAL(7, 75, 15) // "sendFakeUpdates"

    },
    "FakeSimulation\0speedUpdated\0\0newSpeed\0"
    "rpmUpdated\0newRpm\0simulationFinished\0"
    "sendFakeUpdates"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_FakeSimulation[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   34,    2, 0x06 /* Public */,
       4,    1,   37,    2, 0x06 /* Public */,
       6,    0,   40,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       7,    0,   41,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void FakeSimulation::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<FakeSimulation *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->speedUpdated((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->rpmUpdated((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->simulationFinished(); break;
        case 3: _t->sendFakeUpdates(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (FakeSimulation::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&FakeSimulation::speedUpdated)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (FakeSimulation::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&FakeSimulation::rpmUpdated)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (FakeSimulation::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&FakeSimulation::simulationFinished)) {
                *result = 2;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject FakeSimulation::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_FakeSimulation.data,
    qt_meta_data_FakeSimulation,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *FakeSimulation::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *FakeSimulation::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_FakeSimulation.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int FakeSimulation::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void FakeSimulation::speedUpdated(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void FakeSimulation::rpmUpdated(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void FakeSimulation::simulationFinished()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
