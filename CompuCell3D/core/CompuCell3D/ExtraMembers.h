#ifndef EXTRAMEMBERS_H
#define EXTRAMEMBERS_H

#include <climits>
#include <vector>

namespace CompuCell3D {

    /**
	Written by T.J. Sego, Ph.D.
	*/

    // Member creation interface with group factory
    template<class T>
    class ExtraMembersFactory {
    public:
        virtual T *create() = 0;

        virtual void destroy(T *em) = 0;
    };

    // Member factory interface with accessor
    template<class B, class T>
    class ExtraMembersCastor : public ExtraMembersFactory<B> {
    public:
        B *create() { return new T; }

        virtual void destroy(B *em) {
//            delete (B*)em;
            delete static_cast<T *>(em);
        }
    };

    class ExtraMembersGroupFactory;

    // Extra members group; classes can be dynamically assigned per group and instantiated per group instance
    class ExtraMembersGroup {
        std::vector<void *> members;

    public:
        ExtraMembersGroup(const std::vector<ExtraMembersFactory<void> *> &factories) {
            for (auto &itr: factories) members.push_back(itr->create());
        }

        ~ExtraMembersGroup() { destroy(); }

        void *getMember(const unsigned int &_memberIdx) { return members[_memberIdx]; }

        void destroy() {
            for (unsigned int i = 0; i < members.size(); i++) {
                delete (members[i]);
                members[i] = nullptr;
            }
        }
    };

    // Group accessor; each accessor accesses a class instance of a group instance; for use in deployments
    class ExtraMembersGroupAccessorBase {
        unsigned int memberIdx;

    protected:
        void setId(const unsigned int &_id) { memberIdx = _id; }

        virtual ExtraMembersFactory<void> *createFactory() = 0;

        void *getMember(ExtraMembersGroup *g) const { return g->getMember(memberIdx); }

        friend ExtraMembersGroupFactory;

    public:
        ExtraMembersGroupAccessorBase() : memberIdx(UINT_MAX) {};
    };

    // Group accessor; each accessor accesses a class instance of a group instance; for use when specifying an extra member
    template<class T>
    class ExtraMembersGroupAccessor : public ExtraMembersGroupAccessorBase {

    protected:
        ExtraMembersFactory<void> *createFactory() { return new ExtraMembersCastor<void, T>; };
        friend ExtraMembersGroupFactory;

    public:
        // Returns a pointer to the class T instance of the group g
        T *get(ExtraMembersGroup *g) { return (T *) ExtraMembersGroupAccessorBase::getMember(g); };

    };

    // Extra members group factory; creates a group of extra member class instances paired with member accessors
    class ExtraMembersGroupFactory {
        std::vector<ExtraMembersFactory<void> *> factories;

    public:
        ~ExtraMembersGroupFactory() {
            for (unsigned int i = 0; i < factories.size(); i++) {
                delete factories[i];
                factories[i] = 0;
            }
        }

        // Registers a class as an extra member using its accessor
        void registerClass(ExtraMembersGroupAccessorBase *accessor) {
            accessor->setId((unsigned int) factories.size());
            factories.push_back(accessor->createFactory());
        }

        // Creates a new instance of a group
        ExtraMembersGroup *create() { return new ExtraMembersGroup(factories); }

        // Destroys a group
        void destroy(ExtraMembersGroup *g) { g->destroy(); }

    };

};

#endif //EXTRAMEMBERS_H