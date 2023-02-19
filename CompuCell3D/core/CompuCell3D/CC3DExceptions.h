#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <iostream>
#include <string>

namespace CompuCell3D {

    /**
	Written by T.J. Sego, Ph.D.
	*/

    class CC3DException : public std::exception {

        std::string message;
        std::string filename;
        CC3DException *cause=nullptr;

    public:

        CC3DException() {}

        CC3DException(const std::string _message) : message(_message) {}

        CC3DException(const std::string _message, const std::string &_filename) : message(_message),
                                                                                  filename(_filename) {}

        CC3DException(const std::string _message, const CC3DException &_cause) : message(_message) {
            this->cause = new CC3DException(_cause);
        }

        CC3DException(const std::string _message, const std::string &_filename, const CC3DException &_cause) : message(
                _message), filename(_filename) {
            this->cause = new CC3DException(_cause);
        }

        CC3DException(const CC3DException &other) : message(other.message), filename(other.filename),
                                                    cause(other.cause) {}

        // adding std::exception interface implementation
        virtual const char* what() const noexcept {
            return message.c_str();
        }

        virtual ~CC3DException() {
            if (this->cause){
                delete this->cause;
                this->cause = nullptr;
            }
        }

        const std::string getMessage() const { return message; }

        const std::string getFilename() const { return filename; }

        friend std::ostream &operator<<(std::ostream &_os, const CC3DException &_e);
    };

    inline std::ostream &operator<<(std::ostream &_os, const CC3DException &_e) {
        _os << _e.getMessage();
        return _os;
    }

}


#define THROW(msg) throw CC3DException((msg))
#define ASSERT_OR_THROW(msg, condition) {if (!(condition)) THROW(msg);}

#endif // EXCEPTIONS_H