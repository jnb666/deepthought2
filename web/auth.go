package web

import (
	"errors"
	"log"
	"net/http"

	"github.com/goji/httpauth"
	"github.com/gorilla/securecookie"
	"github.com/msteinert/pam"
)

const (
	cookieName  = "deepthought2"
	cookieValue = "authenticated"
)

type AuthMiddleware struct {
	sc   *securecookie.SecureCookie
	opts httpauth.AuthOptions
}

// Setup new middleware for authenticating requests.
func NewAuthMiddleware() AuthMiddleware {
	hashKey := []byte(securecookie.GenerateRandomKey(32))
	blockKey := []byte(securecookie.GenerateRandomKey(32))
	return AuthMiddleware{
		sc:   securecookie.New(hashKey, blockKey),
		opts: httpauth.AuthOptions{Realm: "Restricted", AuthFunc: authPam},
	}
}

// If session cookie is not present then use basic auth + pam to login and set a cookie.
func (mw AuthMiddleware) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if cookie, err := r.Cookie(cookieName); err == nil {
			var value string
			if err = mw.sc.Decode(cookieName, cookie.Value, &value); err == nil && value == cookieValue {
				next.ServeHTTP(w, r)
				return
			}
		}
		httpauth.BasicAuth(mw.opts)(mw.setCookie(next)).ServeHTTP(w, r)
	})
}

func (mw AuthMiddleware) setCookie(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if encoded, err := mw.sc.Encode(cookieName, cookieValue); err == nil {
			cookie := &http.Cookie{Name: cookieName, Value: encoded, Path: "/"}
			http.SetCookie(w, cookie)
		} else {
			log.Println("error encoding cookie:", err)
		}
		h.ServeHTTP(w, r)
	})
}

func authPam(user, pass string, r *http.Request) bool {
	t, err := pam.StartFunc("", "", func(s pam.Style, msg string) (string, error) {
		switch s {
		case pam.PromptEchoOn:
			return user, nil
		case pam.PromptEchoOff:
			return pass, nil
		default:
			return "", errors.New("unexpected style")
		}
	})
	if err != nil {
		log.Println("pam auth error:", err)
		return false
	}
	ok := t.Authenticate(0) == nil
	log.Println("auth", user, ok)
	return ok
}
