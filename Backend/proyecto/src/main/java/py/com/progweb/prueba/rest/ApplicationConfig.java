package py.com.progweb.prueba.rest;

import javax.ws.rs.ApplicationPath;
import javax.ws.rs.GET;
import javax.ws.rs.core.Application;

import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;

@ApplicationPath("/")
public class ApplicationConfig extends Application {
    @GET
    public void reqUser(HttpRequest req, HttpResponse resp) {
        resp.getHeaderNames
    }
}
