FROM nginx:alpine
RUN apk add --no-cache curl
COPY services/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY services/frontend/index.html /usr/share/nginx/html/index.html
COPY services/frontend/style.css /usr/share/nginx/html/style.css
COPY services/frontend/app.js /usr/share/nginx/html/app.js
COPY services/frontend/heroes.json /usr/share/nginx/html/heroes.json
COPY services/frontend/teams.json /usr/share/nginx/html/teams.json
COPY services/frontend/img/ /usr/share/nginx/html/img/
EXPOSE 80
